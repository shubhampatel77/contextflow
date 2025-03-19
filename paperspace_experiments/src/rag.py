import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MistralForCausalLM, AutoTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
import bitsandbytes as bnb
from tqdm.auto import tqdm
from typing import List, Tuple, Dict
from trl import setup_chat_format
from huggingface_hub import HfApi, Repository
from .utils import setup_logger
logger = setup_logger(enable_logging=True)

class RAGSequence(nn.Module):
    """
    RAG implementation supporting decoder-only LLMs with joint optimization.
    
    Implements novel probability marginalization approach:
    p_theta(y^(m) | z_k^(m), x^(m)) approximately equals p_theta(y^(m) | f(z_k^(m), x^(m)))
    
    Args:
        question_encoder: DPR question encoder
        retriever: Document retriever
        generator_tokenizer: Tokenizer for decoder-only LLM
        generator: Decoder-only language model
    """
    
    def __init__(
        self, 
        question_encoder, 
        retriever, 
        generator_tokenizer, 
        generator, 
        accelerator, 
        max_seq_len=512, 
    ):
    
        super().__init__()
        self.question_encoder = question_encoder
        self.retriever = retriever
        self.generator_tokenizer = generator_tokenizer
        self.generator = generator
        self.max_seq_len = max_seq_len
        self.accelerator = accelerator
        self.device = accelerator.device
        
    def generate(
        self, 
        input_ids,
        attention_mask=None,
        num_return_sequences: int = 1, 
        max_new_tokens: int = 128, 
        no_repeat_ngram_size: int = 3, 
        do_sample: bool = False, 
        skip_special_tokens: bool = False,
        num_docs: int = 5, 
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Generate answer and return retrieved context.
        
        Args:
            input_ids: [batch_size, max_seq_len] tokenized input ids using 
                self.retriever.question_encoder.tokenizer [DRPQuestionEncoderTokenizerFast]
            attention_mask: [batch_size, max_seq_len]
            num_return_sequences: Number of sequences to generate
            max_new_tokens: Max new tokens to generate
            num_docs: Number of top contexts to retrieve
            **kwargs: Additional generation kwargs
            
        Returns:
            Dict of (generated_answer, retrieved_context if return_context)
        """
        
        # Returns top num_docs documents
        retrieved_doc_ids, retriever_scores = self.retrieve(
            input_ids, 
            attention_mask,
            num_docs=num_docs
        )
        
        contexts = []
        for b in range(input_ids.shape[0]):
            batch_contexts = []
            for d in range(retrieved_doc_ids.shape[1]):
                doc = self.retriever.index.dataset[int(retrieved_doc_ids[b,d].item())]
                batch_contexts.append(f"Title: {doc['title']}\nText: {doc['text']}")
            contexts.append("\n\n".join(batch_contexts))
        
        # Generate prompts
        prompts = self.inference_prompt(
            input_ids, 
            contexts
        )
    
        generation_kwargs = {
            'max_new_tokens': max_new_tokens,
            'num_return_sequences': num_return_sequences,
            'no_repeat_ngram_size': no_repeat_ngram_size,
            'do_sample': do_sample,
            **kwargs
        }
        
        with self.accelerator.autocast():
            outputs = self.generator.generate(input_ids=prompts, **generation_kwargs)
            
        responses = self.generator_tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=kwargs.get('skip_special_tokens', False)
        )
        
        batch_outputs = []
        for i, resp in enumerate(responses):
            output = {"generated_text": resp, "retrieved_context": contexts[i]}
            batch_outputs.append(output)
        
        return batch_outputs

    # TODO: ability to provide custom context
    def inference_prompt(self, input_ids, contexts, instruction=None, system_prompt=None):
        question = self.retriever.question_encoder_tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        if instruction is None:
            instruction = (
                "Use the given context only if relevant to answer the question.\n"
                "Otherwise, if you don't know, just say \"I don't know.\"\n"
            )
        if system_prompt is None:
            system_prompt = "You are a friendly chatbot who always responds precisely."
        # Handle batch of questions
        batch_prompts = []

        for i in range(input_ids.shape[0]):
            question = self.retriever.question_encoder_tokenizer.decode(
                input_ids[i], 
                skip_special_tokens=True
            )
            
            user_prompt = (
                f"{instruction}\n"
                f"Context:\n{contexts[i]}\n"
                f"Question: {question}\n"
                "Answer:"
            )
            
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # no max_length, since not much variation in it, since context is ~ 100 words
            # and question is < 64 so 64 + 5*100 + 100 (say from instruction + system prompt) < 1024
            prompt = self.generator_tokenizer.apply_chat_template(
                message,
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                padding='max_length',  # see above why
                max_length=1024,
                return_tensors="pt"
            )
            batch_prompts.append(prompt)

        # Stack all prompts into batch
        return torch.cat(batch_prompts, dim=0).to(self.device)  

        
    def prepare_for_sft(self):
        if self.generator_tokenizer.chat_template:
            self.generator_tokenizer.chat_template = None
            
        pad_token = self.generator_tokenizer.pad_token
        vocab_size = self.generator.config.vocab_size
        embedding_dim_before = self.generator.get_input_embeddings().weight.shape
        
        # setup_chat_format() modifies pad token to <|im_end|> so add back pad_token again
        self.generator, self.generator_tokenizer = setup_chat_format(self.generator, self.generator_tokenizer)
        self.generator_tokenizer.pad_token = pad_token
        
        embedding_dim_after = self.generator.get_input_embeddings().weight.shape
        logger.info(f"Embedding matrix before: {embedding_dim_before}, after setup_chat_format(): {embedding_dim_after}")
        
        # resize for [<|im_start|>, <|im_end|>]
        logger.info("Resizing model's embedding matrix vocab_size...")
        self.generator.resize_token_embeddings(len(self.generator_tokenizer))
        logger.info(f"Resized from {vocab_size} to {len(self.generator_tokenizer)}")
        
        # SEE Llama tokenizers and their training strategy
        # self.generator_tokenizer.padding_side = "right"
        
    def add_pad_token(self, pad_token):
        vocab_size = self.generator.config.vocab_size
        
        self.generator_tokenizer.add_tokens([pad_token])
        self.generator_tokenizer.pad_token = pad_token
        # resize for pad_token
        logger.info("Resizing model's embedding matrix vocab_size...")
        self.generator.resize_token_embeddings(len(self.generator_tokenizer))
        logger.info(f"Resized from {vocab_size} to {len(self.generator_tokenizer)}")

    def forward(self, input_ids, attention_mask, labels, debug_print=False):
        # Retrieve documents
        retrieved_doc_ids, retriever_scores = self.retrieve(input_ids, attention_mask)
        
        # Generate sequences for each retrieved document
        logits, labels = self.generate_sequences(input_ids, retrieved_doc_ids, labels, debug_print=debug_print)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.compute_loss(retriever_scores, logits, labels)
        
        return loss
    
    def retrieve(self, input_ids, attention_mask, num_docs=5):
        question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
        docs_dict = self.retriever(
            input_ids.cpu().numpy(),
            question_hidden_states.cpu().detach().numpy(),
            n_docs=num_docs,
            return_tensors="pt"
        )
        retrieved_doc_ids = docs_dict['doc_ids'].to(input_ids.device) # batch_size x num_docs

        # docs_dict['retrieved_doc_embeds'].shape = batch_size x num_docs x embedding_dim (768)
        
        retriever_scores = torch.bmm(
            question_hidden_states.cpu().detach().unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
        ).squeeze(1).to(input_ids.device) # batch size x num_docs
        
        return retrieved_doc_ids, retriever_scores

    def generate_sequences(self, input_ids, retrieved_doc_ids, labels, debug_print=False):
        batch_size, num_docs = retrieved_doc_ids.shape
    
        inputs, attention_mask, labels = self.construct_prompts_and_labels(input_ids, retrieved_doc_ids, labels)
        
        if debug_print:
            # Print shapes to verify
            logger.info(f"Input IDs Shape: {inputs.shape}")
            logger.info(f"Attention Mask Shape: {attention_mask.shape}")
            logger.info(f"Labels Shape: {labels.shape}")
            
            # Decode one sequence for verification (just the first one in batch_size * num_docs)
            for skip_special_tokens in [False, True]:
                logger.info(f"======= SKIP SPECIAL TOKENS: {skip_special_tokens} ==========")
                decoded_prompt = self.generator_tokenizer.decode(inputs[0], skip_special_tokens=skip_special_tokens)
                decoded_label = self.generator_tokenizer.decode(labels[0][labels[0] != -100], skip_special_tokens=skip_special_tokens)
                logger.info(f"Decoded Prompt: {decoded_prompt}")
                logger.info(f"Decoded Label: {decoded_label}")
                
        # Move to the appropriate device (if necessary)
        inputs = inputs.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass through generator, passing labels for loss calculation
        outputs = self.generator(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        
        # Print the model's loss for debugging
        if debug_print:
            logger.info(f"Model Loss (src code): {outputs.loss}")
        
        return logits, labels

    def construct_prompts_and_labels(self, input_ids, retrieved_doc_ids, labels):
        batch_size, num_docs = retrieved_doc_ids.shape
        total_samples = batch_size * num_docs
        tokenized_inputs = torch.zeros((total_samples, self.max_seq_len), dtype=torch.long).to(self.device)

        for i in range(batch_size):
            question = self.retriever.question_encoder_tokenizer.decode(input_ids[i], skip_special_tokens=True)
            for j in range(num_docs):
                doc_id = retrieved_doc_ids[i, j]
                doc = self.retriever.index.dataset[int(doc_id.item())]
                context = f"Title: {doc['title']}\nText: {doc['text']}"
                
                prompt = (
                        "Use the given context only if relevant to answer the question.\n"
                        "If you don't know or if the answer is not in the context, just say \"I don't know.\"\n"
                        f"Context:\n{context}\n"
                        f"Question: {question}\n"
                        f"Answer: {labels[i]}"
                    )
                message = [
                    {"role": "system", "content": "You are a friendly chatbot who always responds precisely."},
                    {"role": "user", "content": prompt}
                ]
                
                # NOTE:
                # 279 is the max prompt length for num_docs = 5, 421 for 10, so generator_tokenizer's 
                # max seq length 512 balances both RAM and undue truncation
                tokenized_inputs[i * num_docs + j] = self.generator_tokenizer.apply_chat_template(
                    message,
                    tokenize=True,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_seq_len
                )

        attention_mask = (tokenized_inputs != self.generator_tokenizer.pad_token_id).long()

        labels = torch.full_like(tokenized_inputs, fill_value=-100)
        answer_token = self.generator_tokenizer.encode("Answer:", add_special_tokens=False)[-1]
        
        for i in range(total_samples):
            answer_start = (tokenized_inputs[i] == answer_token).nonzero(as_tuple=True)[0]
            if len(answer_start) > 0:
                answer_start = answer_start[-1].item() + 1
                labels[i, answer_start:] = tokenized_inputs[i, answer_start:]

        labels = labels.masked_fill(attention_mask == 0, -100)


        return tokenized_inputs, attention_mask, labels

    def compute_loss(self, retriever_scores, logits, labels):

        vocab_size = logits.shape[-1]
        batch_size, num_docs = retriever_scores.shape

        # Shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Reshape tensors
        shift_logits = shift_logits.view(batch_size, num_docs, -1, vocab_size)
        shift_labels = shift_labels.view(batch_size, num_docs, -1)

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Create mask for non-ignored tokens
        mask = (shift_labels != -100)
        
        # Clamp labels to valid range, to prevent CUDA assertion triggers
        # since dim_idx = 0 to vocab_size - 1, and labels should span that
        # hence ignore -100 indexed log_probs
        shift_labels = torch.clamp(shift_labels, 0, vocab_size - 1)

        # Gather target log probabilities
        target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        # Apply mask and compute sequence log probabilities
        masked_log_probs = target_log_probs * mask.float()
        seq_log_probs = masked_log_probs.sum(-1)
        # print(seq_log_probs)

        # Compute generator loss, see notes and torch.CrossEntropy() docs
        # N spans batch_size, num_docs and max_seq_len
        # avg not by N but N - |labels == -100| = mask.sum()
        gen_loss = -masked_log_probs.sum()/mask.sum()
        # print(f"calculated loss = {gen_loss}")

        # Compute retriever log probabilities
        retriever_log_probs = F.log_softmax(retriever_scores, dim=-1)
        # print(retriever_log_probs)
        # Combine sequence and retriever log probabilities
        total_log_probs = seq_log_probs + retriever_log_probs

        # Marginalize over documents
        marginalized_log_probs = torch.logsumexp(total_log_probs, dim=1)

        # Compute final loss (negative log likelihood)
        loss = -marginalized_log_probs.mean()
        # print(f"final computed loss = {loss}")

        return loss
    
    # Other cleanup options like accelerate.clear(), gc.collect(), torch.cuda.ipc_collect()
    # empty CUDA cache, del model don't seem to work, see 03/15 TODO point 3.
    def to_cpu(self):
        """Efficiently move all components to CPU."""
        self.question_encoder.to('cpu')
        self.generator.to('cpu')
        if hasattr(self, 'retriever') and hasattr(self.retriever, 'to'):
            self.retriever.to('cpu')


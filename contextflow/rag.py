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

class RAGSequence(nn.Module):
    """
    RAG implementation supporting decoder-only LLMs with joint optimization.
    
    Implements novel probability marginalization approach:
    p_θ(y^(m) | z_k^(m), x^(m)) ≈ p_θ(y^(m) | f(z_k^(m), x^(m)))
    
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
            input_ids
            attention_mask
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

    def inference_prompt(self, input_ids, retrieved_context, instruction=None, system_prompt=None):
        question = self.retriever.question_encoder_tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        if instruction is None:
            instruction = (
                "Use the given context only if relevant to answer the question.\n"
                "Otherwise, if you don't know, just say \"I don't know.\"\n"
            )
        user_prompt = (
            f"{instruction}\n"
            f"Context:\n{retrieved_context}\n"
            f"Question: {question}\n"
            "Answer:"
        )
        if system_prompt is None:
            system_prompt = "You are a friendly chatbot who always responds precisely."
        
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # print(message)
        tokenized_prompt = self.generator_tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device, dtype=torch.long)    
        
        return tokenized_prompt
        
    def prepare_for_sft(self):
        if self.generator_tokenizer.chat_template:
            self.generator_tokenizer.chat_template = None
        self.generator, self.generator_tokenizer = setup_chat_format(self.generator, self.generator_tokenizer)
        # if sft-only, then add pad token
        if self.generator_tokenizer.pad_token is None:
            self.generator_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        # resize for Union[<pad>, [<pad>, <|im_start|>, <|im_end|>]]
        self.generator.resize_token_embeddings(len(self.generator_tokenizer))
        # SEE Llama tokenizers and their training strategy
        # self.generator_tokenizer.padding_side = "right"

    def forward(self, input_ids, attention_mask, labels):
        # Retrieve documents
        retrieved_doc_ids, retriever_scores = self.retrieve(input_ids, attention_mask)
        
        # Generate sequences for each retrieved document
        logits, labels = self.generate_sequences(input_ids, retrieved_doc_ids, labels)
        
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

    def generate_sequences(self, input_ids, retrieved_doc_ids, labels):
        batch_size, num_docs = retrieved_doc_ids.shape
    
        inputs, attention_mask, labels = self.construct_prompts_and_labels(input_ids, retrieved_doc_ids, labels)
        
        # Print shapes to verify
        # print(f"Input IDs Shape: {inputs.shape}")
        # print(f"Attention Mask Shape: {attention_mask.shape}")
        # print(f"Labels Shape: {labels.shape}")

        # Decode one sequence for verification (just the first one)
        decoded_prompt = self.generator_tokenizer.decode(inputs[0], skip_special_tokens=True)
        decoded_label = self.generator_tokenizer.decode(labels[0][labels[0] != -100], skip_special_tokens=True)
        
        # print(f"Decoded Prompt: {decoded_prompt}")
        # print(f"Decoded Label: {decoded_label}")
        
        # Move to the appropriate device (if necessary)
        inputs = inputs.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass through generator, passing labels for loss calculation
        outputs = self.generator(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        
        # # Print the model's loss for debugging
        # print(f"Model Loss: {outputs.loss}")
        
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


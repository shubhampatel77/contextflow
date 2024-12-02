# update_mistral.py
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DPRQuestionEncoder,
    DataCollatorForLanguageModeling, AutoConfig, PreTrainedModel,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)
from box import Box
import os
import json
import pickle
import bitsandbytes as bnb
from tqdm.auto import tqdm

import shutil
import wandb
import re
from huggingface_hub import upload_file, upload_folder, hf_hub_download, snapshot_download, list_repo_files
import huggingface_hub
import tempfile
from typing import Tuple, List, Dict, Any, Union

from .dataloader import uft_dataloader, sft_dataloader
from .update_retriever import CustomRetriever
from .rag import RAGSequence
from .utils import (
    EarlyStopping, evaluate, load_json, setup_logger, get_latest_checkpoint, HubUploader,
    load_training_state, log_parameter_info, log_grad_norm
)
logger = setup_logger(enable_logging=True)


from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator

def resume_from_checkpoint(
    repo_id: str, 
    experiment_path: str, 
    config: Box,
    retriever_docs: List[Dict],
    accelerator: Accelerator,
) -> Tuple[Union[PreTrainedModel, RAGSequence], Any, Any, Dict]:
    """
    Resume training from latest checkpoint based on experiment type and stage.
    
    Logic flow:
    1. For UFT/SFT only:
        - Check latest checkpoint in experiment_path
        - Resume from that point
    
    2. For combined training:
        - First check if UFT is complete by looking in "uft" subfolder
        - If UFT complete but SFT not started, load UFT checkpoint and start SFT
        - If SFT in progress, resume from latest SFT checkpoint in "sft" subfolder
        - If UFT in progress, resume from latest UFT checkpoint
        
    Returns:
        Tuple of (model, optimizer_state, scheduler_state, progress)
    """
    try:
        exp_type = config.model.experiment.type
        
        # Determine which stage to resume from
        if exp_type == 'combined':
            # Check UFT completion
            uft_checkpoint, uft_epoch = get_latest_checkpoint(
                repo_id, 
                os.path.join(experiment_path, "uft")
            )
            uft_complete = (
                uft_epoch == config.model.training.unsupervised.optimization.num_epochs 
                if uft_checkpoint else False
            )
            
            # Check SFT progress
            sft_checkpoint, sft_epoch = get_latest_checkpoint(
                repo_id,
                os.path.join(experiment_path, "sft")
            )
            
            if sft_checkpoint:  # SFT in progress
                resume_path = os.path.join(experiment_path, "sft")
                latest_checkpoint = sft_checkpoint
                is_supervised = True
            # TODO: Nothing to resume from, its second independent stage of the combined process, 
            # just need to load the appropriate model, hence handle case separately in load or finetune
            elif uft_complete:  # UFT done, start SFT
                resume_path = os.path.join(experiment_path, "uft") 
                latest_checkpoint = uft_checkpoint
                is_supervised = True
            else:  # Resume UFT
                resume_path = os.path.join(experiment_path, "uft")
                latest_checkpoint = uft_checkpoint
                is_supervised = False
                
        else:  # UFT/SFT only
            resume_path = experiment_path
            latest_checkpoint, _ = get_latest_checkpoint(repo_id, resume_path)
            is_supervised = (exp_type == 'sft-only')
        
        # TODO: This case will not occur since it will be taken care of in load or fine tune.
        if not latest_checkpoint:
            logger.info("No checkpoint found to resume from")
            return None, None, None, None
            
        logger.info(f"Resuming from checkpoint: {latest_checkpoint}")

        # Load from latest checkpoint (intermediary epoch)
        resumed_model = load_trained_model(
            repo_id, 
            latest_checkpoint, 
            config, 
            retriever_docs, 
            accelerator,
            do_training=True # always needed whenever laoding a hub ckpt, see load trained model
        )

        
        # Load training state
        if exp_type == 'combined':
            if uft_complete:
                logger.info("Initiating SFT stage for combined")
                return resumed_model, None, None, None, latest_checkpoint, is_supervised
            
        optimizer_state, scheduler_state, progress = load_training_state(
            repo_id=repo_id,
            checkpoint_path=latest_checkpoint
        )
        
        log_parameter_info(resumed_model.question_encoder if is_supervised else None, config.model.retriever.question_encoder)
        log_parameter_info(resumed_model.generator, config.model.generator.base_model)
        
        return base_model, optimizer_state, scheduler_state, progress, latest_checkpoint, is_supervised

    # TODO: This case will not occur since it will be taken care of in load or fine tune.
    except Exception as e:
        logger.error(f"Error resuming from checkpoint: {e}")
        logger.info("Starting fresh training...")
        return None, None, None, None


def train(
    model,
    tokenizer,
    accelerator,
    train_dataloader,
    val_dataloader,
    repo_id,
    experiment_path,
    gradient_accumulation_steps,
    learning_rate,
    num_epochs,
    eval_frequency,
    do_eval,
    weight_decay,
    warmup_ratio,
    max_grad_norm,
    do_early_stopping,
    patience,
    min_delta,
    optimizer_state=None,
    scheduler_state=None,
    progress=None,
    is_supervised=False,
    stage="",
    save_optimizer_and_scheduler=False
):
    uploader = HubUploader()
    
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    optimizer = bnb.optim.AdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )

    # Cosine scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * warmup_ratio), 
        num_training_steps=total_steps
    )

    if optimizer_state and scheduler_state:
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)  
              
    if val_dataloader:
        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)
    else:
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # hard coded project name
    wandb.init(project="rag-optimization", name=f"run_{experiment_path}")
    if progress:
        # TODO: handle resuming logical, loading losses, wandb resuming logic
        # Resume training state
        start_epoch = progress['epoch']
        optimizer_step = progress['optimizer_step']
        all_losses = progress['all_losses']
        epoch_losses = progress['epoch_losses']
        val_losses = progress.get('val_losses', [])
        best_val_loss = min(val_losses) if val_losses else float('inf')
        
        # Resume wandb
        wandb.init(
            project="rag-optimization",
            name=f"run_{experiment_path}",
            resume="allow",
            id=progress.get('wandb_run_id')
        )
        
    else:
        start_epoch = 0
        optimizer_step = 0
        all_losses = []
        best_val_loss = float('inf')   
        val_losses = []
    
    eval_steps = (len(train_dataloader) // gradient_accumulation_steps ) // eval_frequency
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
 
    if is_supervised:
        model.generator.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
    torch.set_grad_enabled(True)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = []
        total_loss = 0
        val_count = 0
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            inputs = {k: (v if is_supervised and k == 'labels' else v.to(accelerator.device)) for k, v in batch.items()}
            with accelerator.autocast():
                outputs = model(**inputs)
                # TODO: make this logic better
                if is_supervised:
                    loss = outputs / gradient_accumulation_steps
                else:
                    loss = outputs.loss / gradient_accumulation_steps
                
            accelerator.backward(loss)
            total_loss += loss.item() * gradient_accumulation_steps
            epoch_losses.append(loss.item() * gradient_accumulation_steps)
            logger.info(f"step {i+1},loss: {loss.item() * gradient_accumulation_steps:.4f}")

            if (i + 1) % gradient_accumulation_steps == 0:
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                accelerator.clip_grad_norm_(trainable_params, max_grad_norm)
                grad_norm = log_grad_norm(model)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                optimizer_step += 1
                # averaging actual loss per batch by grad accumulation, ie loss on which gradient was computed
                # adding .backward(outputs.loss / grad accumulation) for grad_acummulation such losses
                # these are the same
                avg_loss = total_loss / gradient_accumulation_steps
                all_losses.append(avg_loss)
                logger.info(f"optimizer step: {i+1}, loss: {avg_loss:.4f}, grad norm: {grad_norm:.4f}")
                wandb.log({"train_loss": avg_loss, "learning_rate": scheduler.get_last_lr()[0], "grad_norm": grad_norm})
                total_loss = 0
                
                if optimizer_step % eval_steps == 0 and do_eval:
                    val_loss = evaluate(model, val_dataloader, epoch, val_count, accelerator, is_supervised)
                    model.train() # back to train
                    val_losses.append(val_loss)
                    val_count += 1
                    logger.info(f"global step {optimizer_step}: validation Loss = {val_loss:.4f}")
                    wandb.log({"val_loss": val_loss})
                    
                    # TODO: hardcoded min_epoch before saving best val model
                    if val_loss < best_val_loss and epoch >= 5:
                        best_val_loss = val_loss
                        ckpt_name = f"best_model_epoch_{epoch+1}_step_{optimizer_step}"
                        progress = {
                            'wandb_run_id': wandb.run.id,
                            'epoch': epoch,
                            'optimizer_step': optimizer_step, 
                            'all_losses': all_losses, 
                            'epoch_losses': epoch_losses, 
                            'val_losses': val_losses, 
                        }
                        checkpoint = {
                            'generator': {
                                'model': model.generator if is_supervised else model,
                                'tokenizer': tokenizer,
                                'training_state': {
                                    'optimizer': optimizer,
                                    'scheduler': scheduler,
                                    'progress': progress
                                }
                            }
                        }

                        if is_supervised:
                            checkpoint['retriever'] = {
                                'question_encoder': model.question_encoder
                            }
                        # stage is "" when not combined
                        path_in_repo = os.path.join(experiment_path, stage, "val", ckpt_name)

                        uploader.upload_to_hub(
                            content=checkpoint,
                            path_in_repo=path_in_repo,
                            repo_id=repo_id
                        )                            
                 
                    if early_stopping(val_loss) and do_early_stopping:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}, step {optimizer_step}")
                        return

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"epoch {epoch + 1}: average training loss = {avg_train_loss:.4f}")
        wandb.log({"epoch_train_loss": avg_train_loss, "epoch": epoch + 1})
        
        ckpt_name = f"checkpoint-epoch-{epoch+1}"
        progress = {
            'wandb_run_id': wandb.run.id,
            'epoch': epoch,
            'optimizer_step': optimizer_step, 
            'all_losses': all_losses, 
            'epoch_losses': epoch_losses, 
            'val_losses': val_losses
        }
        
        
        checkpoint = {
            'generator': {
                'model':  model.generator if is_supervised else model,
                'tokenizer': tokenizer,
                'training_state': {
                    'optimizer': optimizer,
                    'scheduler': scheduler,
                    'progress': progress
                }
            }
        }

        if is_supervised:
            checkpoint['retriever'] = {
                'question_encoder': model.question_encoder
            }

        path_in_repo = os.path.join(experiment_path, stage, ckpt_name)

        uploader.upload_to_hub(
            content=checkpoint,
            path_in_repo=path_in_repo,
            repo_id=repo_id
        )
    wandb.finish()
    del model
    del optimizer
    del scheduler
    torch.cuda.empty_cache()
    name = "End-to-end supervised FT" if is_supervised else "Unsupervised FT"
    logger.info(f"{name} completed.")
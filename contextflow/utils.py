from datetime import datetime
import pytz
from dateutil import parser
import os
import logging
import json
import torch
from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.auto import tqdm
from typing import Union, Optional, Any, Tuple, Dict
from pathlib import Path
import shutil
import tempfile
import re

from pathlib import Path
from datasets import Dataset
import faiss
import yaml
from box import Box
import yaml
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import PeftModelForCausalLM
from .rag import RAGSequence
from peft import PeftModel
from huggingface_hub import list_repo_files, upload_file, upload_folder

def setup_logger(log_dir='logs', log_level=logging.INFO, enable_logging=False):
    if not enable_logging:
        logging.disable(logging.CRITICAL)
        return None

    # Get the logger instance
    logger = logging.getLogger(__name__)

    # Check if the logger already has handlers (to prevent duplicate logging)
    if not logger.hasHandlers():
        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a unique log file name
        log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Set up basic configuration for logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # This will print logs to console as well
            ]
        )
        
        logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

logger = setup_logger(enable_logging=True)

# utils for processing documents
def parse_date(date_str):
    if not date_str or date_str == "1":
        return None
    try:
        if len(date_str) <= 4 and date_str.isdigit():  # Year only
            return datetime(int(date_str), 1, 1, tzinfo=pytz.UTC)
        else:  # 3-letter-month Year
            return datetime.strptime(date_str, "%b %Y").replace(tzinfo=pytz.UTC)
    except ValueError:
        try:
            return parser.parse(date_str).replace(tzinfo=pytz.UTC)
        except (ValueError, TypeError):
            print(f"Warning: Unable to parse date string: {date_str}")
            return None       
        
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def split_text(text: str, n=100, character=" "):
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]

def load_experiment_config(repo_id: str, experiment_path: str) -> Box:
    """
    Download and load experiment config from hub.
    
    Args:
        repo_id: HuggingFace repo ID
        experiment_path: Path to experiment directory
    
    Returns:
        Box: Loaded config object
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # hardcoded config.yml
            config_path = os.path.join(experiment_path, "config.yml")
            local_config = hf_hub_download(
                repo_id=repo_id,
                filename=config_path,
                local_dir=temp_dir
            )
            
            with open(local_config, 'r') as f:
                config = Box(yaml.safe_load(f))
                
            return config
            
        except Exception as e:
            logger.error(f"Error loading config from hub: {e}")
            raise


class HubUploader:
    """Handler for uploading content to HuggingFace Hub with proper path handling."""
    
    @staticmethod
    def upload_to_hub(
        content: Union[Dict, Any],
        path_in_repo: str,
        repo_id: str,
        commit_message: Optional[str] = None
    ) -> None:
        """
        Upload content to HuggingFace Hub handling nested structures and maintaining paths.
        
        Args:
            content: Content to upload (Dict or supported type)
            path_in_repo: Base path in repo
            repo_id: HuggingFace repo ID
            commit_message: Optional commit message
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_base = Path(temp_dir)
            
            try:
                if isinstance(content, dict) and not isinstance(content, Box):
                    # For dict content, save each component and upload entire folder
                    for key, value in content.items():
                        HubUploader._save_with_path(
                            value, 
                            temp_base, 
                            path_in_repo,
                            key
                        )
                    
                    # Upload entire directory
                    upload_folder(
                        folder_path=str(temp_base),
                        path_in_repo=path_in_repo,
                        repo_id=repo_id,
                        commit_message=commit_message or f"Upload to {path_in_repo}"
                    )
                else:
                    # For single objects, use direct save and upload
                    save_path = HubUploader._save_content(content, temp_base)
                    if save_path.is_dir():
                        upload_folder(
                            folder_path=str(save_path),
                            path_in_repo=path_in_repo,
                            repo_id=repo_id,
                            commit_message=commit_message or f"Upload to {path_in_repo}"
                        )
                    else:
                        upload_file(
                            path_or_fileobj=str(save_path),
                            path_in_repo=path_in_repo,
                            repo_id=repo_id,
                            commit_message=commit_message or f"Upload to {path_in_repo}"
                        )
                
                logger.info(f"Successfully uploaded to {repo_id}/{path_in_repo}")
                
            except Exception as e:
                logger.error(f"Failed to upload content: {e}")
                raise

    @staticmethod
    def _save_with_path(content: Any, base_path: Path, repo_path: str, key: str) -> None:
        """Save content maintaining directory structure and handling nested dicts."""
        if isinstance(content, dict):
            # Create directory for nested dict
            current_path = base_path / key
            os.makedirs(current_path, exist_ok=True)
            
            # Handle special cases for training state
            if key == 'training_state':
                for state_key, state_value in content.items():
                    if state_key == 'optimizer':
                        torch.save(state_value.state_dict(), current_path / 'optimizer.pt')
                    elif state_key == 'scheduler':
                        torch.save(state_value.state_dict(), current_path / 'scheduler.pt')
                    elif state_key == 'progress':
                        torch.save(state_value, current_path / 'progress.pt')
                    else:
                        save_path = HubUploader._save_content(state_value, current_path / state_key)
            else:
                # Regular nested dict handling
                for subkey, subvalue in content.items():
                    HubUploader._save_with_path(
                        subvalue,
                        current_path,
                        os.path.join(repo_path, key),
                        subkey
                    )
        else:
            save_path = HubUploader._save_content(content, base_path / key)

    @staticmethod
    def _save_content(content: Any, path: Path) -> Path:
        """Save individual content items with appropriate methods."""
        os.makedirs(path.parent, exist_ok=True)
        
        if isinstance(content, Dataset):
            os.makedirs(path, exist_ok=True)
            content.save_to_disk(str(path))
            return path
            
        elif isinstance(content, faiss.Index):
            faiss_path = path.with_suffix('.faiss')
            faiss.write_index(content, str(faiss_path))
            return faiss_path
            
        elif isinstance(content, Box):
            yaml_path = path.with_suffix('.yml')
            with open(yaml_path, 'w') as f:
                yaml.dump(content.to_dict(), f, default_flow_style=False)
            return yaml_path
            
        elif hasattr(content, 'save_pretrained'):
            # to ensure that model's config is saved as PEFT wrapper only stores adapter weights
            # it does not store any changes in vocab size
            os.makedirs(path, exist_ok=True)
            if isinstance(content, PeftModelForCausalLM):
                base_config = content.get_base_model().config.to_dict()
                with open(path / "base_config.json", 'w') as f:
                    json.dump(base_config, f)
            content.save_pretrained(path)
            return path
            
        elif isinstance(content, (Optimizer, _LRScheduler)):
            state_path = path.with_suffix('.pt')
            torch.save(content.state_dict(), state_path)
            return state_path
            
        elif isinstance(content, (dict, list)):
            data_path = path.with_suffix('.pt')
            torch.save(content, data_path)
            return data_path
            
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
    

def get_latest_checkpoint(repo_id: str, experiment_path: str) -> Tuple[str, int]:
    """
    Get the latest checkpoint path and epoch number from HuggingFace repository.
    
    Args:
        repo_id: HuggingFace repository ID
        experiment_path: Path to experiment directory
        
    Returns:
        tuple: (checkpoint_path, epoch_number)
    """
    # Match only the epoch number
    checkpoint_pattern = re.compile(rf"{experiment_path}/checkpoint-epoch-(\d+)(?:/.*)?$")
    
    # Get unique checkpoint paths by removing file-specific suffixes
    checkpoint_epochs = set()
    for file in list_repo_files(repo_id):
        if match := checkpoint_pattern.match(file):
            checkpoint_epochs.add(int(match.group(1)))
    
    if not checkpoint_epochs:
        return None, None
        
    latest_epoch = max(checkpoint_epochs)
    latest_checkpoint = f"{experiment_path}/checkpoint-epoch-{latest_epoch}"
    
    return latest_checkpoint, latest_epoch

def load_training_state(repo_id: str, checkpoint_path: str) -> Tuple[Optional[dict], Optional[dict], Optional[dict]]:
    """
    Load training state (optimizer, scheduler, progress) from checkpoint.
    
    Args:
        repo_id: HuggingFace repo ID
        checkpoint_path: Path to checkpoint directory
    
    Returns:
        Tuple of (optimizer_state, scheduler_state, progress)
    """
    try:
        progress_path = os.path.join(checkpoint_path, "progress.pt")
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
        
        progress = None
        optimizer_state = None
        scheduler_state = None
        
        # Try loading progress
        if huggingface_hub.file_exists(repo_id=repo_id, filename=progress_path):
            progress_file = hf_hub_download(repo_id=repo_id, filename=progress_path)
            progress = torch.load(progress_file)
            logger.info(f"Loaded training progress: {progress}")
        
        # Try loading optimizer state
        if huggingface_hub.file_exists(repo_id=repo_id, filename=optimizer_path):
            optimizer_file = hf_hub_download(repo_id=repo_id, filename=optimizer_path)
            optimizer_state = torch.load(optimizer_file)
            logger.info("Loaded optimizer state")
        
        # Try loading scheduler state
        if huggingface_hub.file_exists(repo_id=repo_id, filename=scheduler_path):
            scheduler_file = hf_hub_download(repo_id=repo_id, filename=scheduler_path)
            scheduler_state = torch.load(scheduler_file)
            logger.info("Loaded scheduler state")
            
        return optimizer_state, scheduler_state, progress
        
    except Exception as e:
        logger.error(f"Error loading training state: {e}")
        return None, None, None

def log_parameter_info(model, name=None):
    if name:
        logger.info(f"====== {name} parameters ======")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params}")
    logger.info(f"Frozen parameters: {frozen_params}")
    
def log_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5  
    return total_norm

# evaluation and early stopping
def evaluate(model, dataloader, epoch, val_count, accelerator, is_supervised):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Validation {epoch+1}.{val_count+1}")):
            inputs = {k: (v if is_supervised and k == 'labels' else v.to(accelerator.device)) for k, v in batch.items()}
            with accelerator.autocast():
                outputs = model(**inputs)
                if is_supervised:
                    loss = outputs
                else:
                    loss = outputs.loss
                total_loss += loss.item()
    model.to(accelerator.device)
    return total_loss / len(dataloader)


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
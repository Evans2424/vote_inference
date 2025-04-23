"""
Vote Identification Model Training

This script trains a token classification model based on BERTimbau to identify votes,
voters, and voting results in deliberation texts.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # Import AdamW from torch.optim instead of transformers
from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BertTokenizerFast
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from seaborn import heatmap
import random
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load tag mapping
try:
    with open('tag_mapping.json', 'r', encoding='utf-8') as f:
        TAG_MAPPING = json.load(f)
    # Create inverse mapping
    ID_TO_TAG = {int(v): k for k, v in TAG_MAPPING.items()}
except FileNotFoundError:
    logger.warning("Tag mapping file not found. Using default mapping.")
    # Default mapping will be used
    TAG_MAPPING = {}
    ID_TO_TAG = {}


class Config:
    """Configuration for model training."""
    # Model parameters
    MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'  # BERTimbau base
    MAX_LENGTH = 512  # Max sequence length
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    EPOCHS = 10
    WARMUP_STEPS = 500
    DROPOUT = 0.1
    EARLY_STOPPING_PATIENCE = 3
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_CLASSES = len(TAG_MAPPING) if TAG_MAPPING else 19  # Default to 19 classes if mapping not found
    
    # Paths
    TRAIN_FILE = 'vote_identification_train.jsonl'
    VAL_FILE = 'vote_identification_val.jsonl'
    TEST_FILE = 'vote_identification_test.jsonl'
    OUTPUT_DIR = 'vote_identifier_model'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VoteIdentificationDataset(Dataset):
    """Dataset for vote identification token classification."""
    
    def __init__(self, file_path, tokenizer, max_length=512):
        """Initialize dataset from JSONL file."""
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Return a preprocessed example at the given index."""
        example = self.examples[idx]
        
        # Get the text and tags
        text = example['text']
        tokens = example['tokens']
        tag_ids = example['tag_ids']
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Align the tags with the tokenized input
        labels = self._align_labels(encoding, tokens, tag_ids)
        
        # Create comprehensive metadata dict with votes, party_collective_references and party_to_participants
        metadata = {
            'votes': example.get('votes', []),
            'party_collective_references': example.get('party_collective_references', []),
            'party_to_participants': example.get('party_to_participants', {}),
            'gendered_party_references': example.get('gendered_party_references', [])
        }
        
        # Return as tensors
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.long),
            'metadata': metadata
        }
    
    def _align_labels(self, encoding, tokens, tag_ids):
        """
        Align the original labels with the tokenized text.
        
        Args:
            encoding: The output from tokenizer
            tokens: Original tokens
            tag_ids: Original tag IDs
            
        Returns:
            Aligned labels for the tokenized text
        """
        # Initialize aligned labels with -100 (ignored in loss calculation)
        aligned_labels = [-100] * len(encoding['input_ids'][0])
        
        # Get word IDs - which tokens correspond to which original word
        word_ids = encoding.word_ids()
        
        # Map original tags to tokenized text
        prev_word_idx = None
        for i, word_idx in enumerate(word_ids):
            # Special tokens have word_idx = None
            if word_idx is None:
                aligned_labels[i] = -100
            # For the first token of a word, use the original label
            elif word_idx != prev_word_idx:
                try:
                    aligned_labels[i] = tag_ids[word_idx]
                except IndexError:
                    # In case word_idx is out of bounds
                    aligned_labels[i] = -100
            # For subsequent tokens of a word, also use the original label
            else:
                try:
                    # Convert B- to I- for subword tokens
                    label = tag_ids[word_idx]
                    if ID_TO_TAG[label].startswith('B-'):
                        # Get the equivalent I- tag
                        i_tag = 'I-' + ID_TO_TAG[label][2:]
                        if i_tag in TAG_MAPPING:
                             aligned_labels[i] = TAG_MAPPING[i_tag]
                        else:
                             aligned_labels[i] = label
                    else:
                         aligned_labels[i] = label
                except IndexError:
                    aligned_labels[i] = -100
            
            prev_word_idx = word_idx
        
        return aligned_labels


class VoteIdentificationTrainer:
    """Trainer for the vote identification model."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.device = config.DEVICE
        self.tokenizer = None
        self.model = None
        
        # Create output directory
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    def setup_model(self):
        """Set up tokenizer and model."""
        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.config.MODEL_NAME}")
        self.tokenizer = BertTokenizerFast.from_pretrained(self.config.MODEL_NAME)
        
        # Load model
        logger.info(f"Loading model: {self.config.MODEL_NAME}")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.MODEL_NAME,
            num_labels=self.config.NUM_CLASSES,
            hidden_dropout_prob=self.config.DROPOUT,
            attention_probs_dropout_prob=self.config.DROPOUT
        )
        self.model.to(self.device)
    
    def train(self):
        """Train the model with early stopping."""
        # Set up model
        self.setup_model()
        
        # Create datasets
        train_dataset = VoteIdentificationDataset(
            self.config.TRAIN_FILE, 
            self.tokenizer, 
            self.config.MAX_LENGTH
        )
        val_dataset = VoteIdentificationDataset(
            self.config.VAL_FILE, 
            self.tokenizer, 
            self.config.MAX_LENGTH
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        # Set up optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Set up learning rate scheduler
        total_steps = len(train_loader) * self.config.EPOCHS // self.config.GRADIENT_ACCUMULATION_STEPS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_f1 = 0.0
        early_stopping_counter = 0
        
        train_losses = []
        val_losses = []
        val_f1_scores = []
        
        for epoch in range(self.config.EPOCHS):
            # Train
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            train_losses.append(train_loss)
            
            # Evaluate
            val_loss, val_metrics = self._evaluate(val_loader)
            val_losses.append(val_loss)
            val_f1_scores.append(val_metrics['macro_f1'])
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Val Metrics: accuracy={val_metrics['accuracy']:.4f}, "
                      f"macro_precision={val_metrics['macro_precision']:.4f}, "
                      f"macro_recall={val_metrics['macro_recall']:.4f}, "
                      f"macro_f1={val_metrics['macro_f1']:.4f}")
            
            # Check for improvement
            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                early_stopping_counter = 0
                
                # Save the best model
                self._save_model(f"{self.config.OUTPUT_DIR}/best_model")
                logger.info(f"  New best model saved with F1: {best_val_f1:.4f}")
            else:
                early_stopping_counter += 1
                logger.info(f"  No improvement. Early stopping counter: {early_stopping_counter}/{self.config.EARLY_STOPPING_PATIENCE}")
            
            # Early stopping
            if early_stopping_counter >= self.config.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered")
                break
        
        # Create visualizations
        self._create_training_plots(train_losses, val_losses, val_f1_scores)
        
        # Evaluate on test set
        self._test_final_model()
    
    def _train_epoch(self, dataloader, optimizer, scheduler):
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        
        # Accumulate gradients over multiple batches
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Normalize loss for gradient accumulation
            loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass
            loss.backward()
            
            # Update parameters every n steps
            if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar and loss
            total_loss += loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS
            progress_bar.set_postfix({'loss': loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS})
        
        return total_loss / len(dataloader)
    
    def _evaluate(self, dataloader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Update loss
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=2)
                
                # Collect predictions and labels (ignoring padding tokens)
                for i in range(len(labels)):
                    for j in range(len(labels[i])):
                        if labels[i][j] != -100:  # Ignore padding tokens
                            all_predictions.append(predictions[i][j].item())
                            all_labels.append(labels[i][j].item())
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions)
        
        return total_loss / len(dataloader), metrics
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=range(self.config.NUM_CLASSES)
        )
        
        # Map class IDs to tag names for reporting
        class_metrics = {}
        for i in range(len(class_precision)):
            if i in ID_TO_TAG:
                tag = ID_TO_TAG[i]
                class_metrics[tag] = {
                    'precision': class_precision[i],
                    'recall': class_recall[i],
                    'f1': class_f1[i]
                }
        
        return {
            'accuracy': accuracy,
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1,
            'per_class': class_metrics
        }
    
    def _save_model(self, output_dir):
        """Save model and tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        config_path = os.path.join(output_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            # Convert config to dict, excluding non-serializable attributes
            config_dict = {k: v for k, v in vars(self.config).items() 
                          if not k.startswith('__') and not callable(v) and k != 'DEVICE'}
            json.dump(config_dict, f, indent=2)
    
    def _create_training_plots(self, train_losses, val_losses, val_f1_scores):
        """Create training visualization plots."""
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(12, 4))
        
        # Plot training curves
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot F1 score progression
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_f1_scores, 'g-')
        plt.title('Validation F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        
        plt.tight_layout()
        plt.savefig(f"{self.config.OUTPUT_DIR}/training_curves.png")
        plt.close()
    
    def _test_final_model(self):
        """Evaluate the final model on the test set."""
        # Load test dataset
        test_dataset = VoteIdentificationDataset(
            self.config.TEST_FILE, 
            self.tokenizer, 
            self.config.MAX_LENGTH
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        # Evaluate
        test_loss, test_metrics = self._evaluate(test_loader)
        
        # Log results
        logger.info("Test Results:")
        logger.info(f"  Test Loss: {test_loss:.4f}")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Macro Precision: {test_metrics['macro_precision']:.4f}")
        logger.info(f"  Macro Recall: {test_metrics['macro_recall']:.4f}")
        logger.info(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
        
        # Print per-class metrics for non-O tags
        logger.info("Per-class metrics:")
        for tag, metrics in test_metrics['per_class'].items():
            if tag != "O":  # Skip the outside tag
                logger.info(f"  {tag}: precision={metrics['precision']:.4f}, "
                          f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}")
        
        # Evaluate party vote detection specifically
        self._evaluate_party_vote_detection(test_loader)
        
        # Save test results
        results_file = f"{self.config.OUTPUT_DIR}/test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        # Create confusion matrix visualization
        self._create_confusion_matrix(test_loader)
    
    def _evaluate_party_vote_detection(self, dataloader):
        """
        Evaluate how well the model detects party vote patterns.
        This focuses on party collective references and VOTANTE tags.
        """
        self.model.eval()
        
        # Track VOTANTE-related predictions
        votante_true_positives = 0
        votante_false_positives = 0
        votante_false_negatives = 0
        
        # Track position linkages to party collective references
        position_linkage_correct = 0
        position_linkage_incorrect = 0
        
        # Tag IDs for relevant entity types
        votante_tag_ids = [tag_id for tag, tag_id in TAG_MAPPING.items() if "VOTANTE" in tag]
        position_tag_ids = [tag_id for tag, tag_id in TAG_MAPPING.items() if "LIGACAO_POSICIONAMENTO" in tag]
        
        # When specific position tag IDs are needed
        favor_tag_ids = [tag_id for tag, tag_id in TAG_MAPPING.items() if "POSICIONAMENTO_FAVOR" in tag]
        contra_tag_ids = [tag_id for tag, tag_id in TAG_MAPPING.items() if "POSICIONAMENTO_CONTRA" in tag]
        abstencao_tag_ids = [tag_id for tag, tag_id in TAG_MAPPING.items() if "POSICIONAMENTO_ABSTENCAO" in tag]
        
        # Track collective reference detection
        collective_refs_matched = 0
        collective_refs_total = 0
        
        # Track context usage ability (when party_to_participants mapping is available)
        context_used_correctly = 0
        context_available_cases = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating vote detection", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                metadata = batch['metadata']
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=2)
                
                # Analyze predictions for each example in the batch
                for i in range(len(predictions)):
                    # Count VOTANTE tag metrics
                    for j in range(len(predictions[i])):
                        if labels[i][j] in votante_tag_ids:
                            if predictions[i][j] == labels[i][j]:
                                votante_true_positives += 1
                            else:
                                votante_false_negatives += 1
                        elif predictions[i][j] in votante_tag_ids:
                            votante_false_positives += 1
                    
                    # Analyze party collective references
                    example_collective_refs = metadata[i]['party_collective_references']
                    party_to_participants = metadata[i]['party_to_participants']
                    
                    if example_collective_refs:
                        collective_refs_total += len(example_collective_refs)
                        
                        # For each detected party collective reference, check if the model correctly 
                        # identified both the VOTANTE entity and linked it with the correct position
                        for ref in example_collective_refs:
                            party_name = ref.get('partido', '')
                            vote_type = ref.get('tipo', '')
                            ref_text = ref.get('text', '')
                            
                            # Check if this party has mapped participants in the context
                            has_participant_mapping = party_name in party_to_participants and len(party_to_participants[party_name]) > 0
                            
                            if has_participant_mapping:
                                context_available_cases += 1
                            
                            # Determine if the model has recognized this collective reference
                            votante_detected = False
                            position_detected = False
                            correct_position_detected = False
                            
                            for j in range(len(predictions[i])):
                                if predictions[i][j] in votante_tag_ids:
                                    votante_detected = True
                                
                                # Check if the position is detected and it's the right type
                                if predictions[i][j] in position_tag_ids:
                                    position_detected = True
                                    
                                    # Check if it's the correct position type
                                    if (vote_type == 'favor' and predictions[i][j] in favor_tag_ids) or \
                                       (vote_type == 'contra' and predictions[i][j] in contra_tag_ids) or \
                                       (vote_type == 'abstencao' and predictions[i][j] in abstencao_tag_ids):
                                        correct_position_detected = True
                            
                            # Credit the model for identifying the collective reference
                            if votante_detected:
                                collective_refs_matched += 1
                                
                                # Check if position was also detected correctly
                                if correct_position_detected:
                                    position_linkage_correct += 1
                                elif position_detected:  # Wrong position detected
                                    position_linkage_incorrect += 1
                                
                                # Check if the model is correctly using context when available
                                # A more sophisticated check would involve seeing if the model can link
                                # the collective reference to individual participants
                                if has_participant_mapping and correct_position_detected:
                                    context_used_correctly += 1
        
        # Calculate metrics for VOTANTE tag detection
        votante_precision = votante_true_positives / (votante_true_positives + votante_false_positives) if (votante_true_positives + votante_false_positives) > 0 else 0
        votante_recall = votante_true_positives / (votante_true_positives + votante_false_negatives) if (votante_true_positives + votante_false_negatives) > 0 else 0
        votante_f1 = 2 * votante_precision * votante_recall / (votante_precision + votante_recall) if (votante_precision + votante_recall) > 0 else 0
        
        # Calculate collective reference detection metrics
        collective_ref_recall = collective_refs_matched / collective_refs_total if collective_refs_total > 0 else 0
        
        # Calculate position linkage accuracy
        position_linkage_accuracy = position_linkage_correct / (position_linkage_correct + position_linkage_incorrect) if (position_linkage_correct + position_linkage_incorrect) > 0 else 0
        
        # Calculate context usage accuracy
        context_usage_accuracy = context_used_correctly / context_available_cases if context_available_cases > 0 else 0
        
        # Log party vote detection metrics
        logger.info("\nVote Detection Metrics:")
        logger.info(f"  VOTANTE Tag Precision: {votante_precision:.4f}")
        logger.info(f"  VOTANTE Tag Recall: {votante_recall:.4f}")
        logger.info(f"  VOTANTE Tag F1: {votante_f1:.4f}")
        logger.info(f"  Collective Reference Detection Recall: {collective_ref_recall:.4f}")
        logger.info(f"  Position Linkage Accuracy: {position_linkage_accuracy:.4f}")
        logger.info(f"  Context Usage Accuracy: {context_usage_accuracy:.4f}")
        logger.info(f"  Total VOTANTE Mentions: {votante_true_positives + votante_false_negatives}")
        logger.info(f"  Total Collective References: {collective_refs_total}")
        logger.info(f"  Total Context Usage Opportunities: {context_available_cases}")
        
        # Save vote detection metrics
        vote_metrics = {
            "votante_tag_precision": votante_precision,
            "votante_tag_recall": votante_recall,
            "votante_tag_f1": votante_f1,
            "collective_ref_recall": collective_ref_recall,
            "position_linkage_accuracy": position_linkage_accuracy,
            "context_usage_accuracy": context_usage_accuracy,
            "votante_true_positives": votante_true_positives,
            "votante_false_positives": votante_false_positives,
            "votante_false_negatives": votante_false_negatives,
            "collective_refs_matched": collective_refs_matched,
            "collective_refs_total": collective_refs_total,
            "context_used_correctly": context_used_correctly,
            "context_available_cases": context_available_cases
        }
        
        # Save vote metrics
        vote_metrics_file = f"{self.config.OUTPUT_DIR}/vote_detection_metrics.json"
        with open(vote_metrics_file, 'w') as f:
            json.dump(vote_metrics, f, indent=2)
        
        return vote_metrics
    
    def _create_confusion_matrix(self, dataloader):
        """Create and save a confusion matrix visualization."""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Creating confusion matrix", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=2)
                
                for i in range(len(labels)):
                    for j in range(len(labels[i])):
                        if labels[i][j] != -100:  # Ignore padding tokens
                            all_predictions.append(predictions[i][j].item())
                            all_labels.append(labels[i][j].item())
        
        # Get unique classes that actually appear in the data
        unique_classes = sorted(list(set(all_labels + all_predictions)))
        
        # Create confusion matrix
        conf_matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
        
        for true, pred in zip(all_labels, all_predictions):
            true_idx = unique_classes.index(true)
            pred_idx = unique_classes.index(pred)
            conf_matrix[true_idx][pred_idx] += 1
        
        # Create class labels
        class_labels = [ID_TO_TAG.get(idx, str(idx)) for idx in unique_classes]
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        heatmap(conf_matrix, annot=True, fmt="d", xticklabels=class_labels, 
                yticklabels=class_labels, cmap="YlGnBu")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{self.config.OUTPUT_DIR}/confusion_matrix.png")
        plt.close()


class EarlyStopping:
    """Early stopping implementation to prevent overfitting."""
    
    def __init__(self, patience=3, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose
    
    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
        
        return self.early_stop


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Train a vote identification model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model", type=str, default="neuralmind/bert-base-portuguese-cased", 
                        help="Pretrained model to use")
    parser.add_argument("--output_dir", type=str, default="vote_identifier_model", 
                        help="Output directory for the trained model")
    
    args = parser.parse_args()
    
    # Update config with command-line arguments
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.MODEL_NAME = args.model
    config.OUTPUT_DIR = args.output_dir
    
    # Train model
    trainer = VoteIdentificationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
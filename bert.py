import pandas as pd
import numpy as np
import json
import time
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold  # Import StratifiedKFold

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load train and test datasets
train_data = pd.read_csv('./data/train_data.csv')
test_data = pd.read_csv('./data/test_data.csv')

# Combine train and test data for cross-validation
data = pd.concat([train_data, test_data], ignore_index=True)

# Extract texts and labels
texts = data['abstract'].tolist()
labels = data['label'].tolist()

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a custom Dataset class
class ReviewDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        text = str(self.reviews[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True,
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc='Training'):
        optimizer.zero_grad()
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def eval_model(model, data_loader):
    model.eval()
    total_loss = 0
    pred_labels = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return total_loss / len(data_loader), pred_labels, true_labels

# Initialize StratifiedKFold
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store metrics
accuracy_list = []
precision_list = []
recall_list = []
f1_score_list = []
evaluation_time_list = []

# Initialize a list to store metrics for each fold
fold_metrics_list = []

# Class names
class_names = [0, 1]

# Start cross-validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(texts, labels)):
    print(f'\nFold {fold + 1}/{n_splits}')
    start_time = time.time()

    # Split data
    X_train_texts = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]

    X_val_texts = [texts[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]

    # Create datasets
    train_dataset = ReviewDataset(X_train_texts, y_train, tokenizer, max_length=128)
    val_dataset = ReviewDataset(X_val_texts, y_val, tokenizer, max_length=128)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize the model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3  # Assuming 3 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training
    for epoch in range(3):  # Number of epochs
        print(f"Epoch {epoch + 1}/3")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"Train loss: {train_loss}")

    # Evaluation
    val_loss, y_pred, y_true = eval_model(model, val_loader)
    print(f"Validation loss: {val_loss}")

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0
    )

    evaluation_time = time.time() - start_time

    # Store metrics
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_score_list.append(f1_score)
    evaluation_time_list.append(evaluation_time)

    # Store metrics for the current fold
    fold_metrics = {
        'fold': fold + 1,
        'accuracy': accuracy,
        'evaluation_time': evaluation_time,
        'metrics_per_class': [
            {
                'class': str(class_names[idx]),
                'precision': precision[idx],
                'recall': recall[idx],
                'f1_score': f1_score[idx],
            }
            for idx in range(len(class_names))
        ]
    }
    fold_metrics_list.append(fold_metrics)

    # Free up memory
    del model
    torch.cuda.empty_cache()

# Calculate average metrics
avg_accuracy = np.mean(accuracy_list)
avg_precision = np.mean(precision_list, axis=0)
avg_recall = np.mean(recall_list, axis=0)
avg_f1_score = np.mean(f1_score_list, axis=0)
total_evaluation_time = np.sum(evaluation_time_list)

# Prepare overall metrics dictionary
overall_metrics = {
    "time": total_evaluation_time,
    'average_accuracy': avg_accuracy,
    'average_metrics_per_class': [],
    'fold_metrics': fold_metrics_list  # Include per-fold metrics
}

for idx, class_name in enumerate(class_names):
    class_metric = {
        'class': str(class_name),
        'precision': avg_precision[idx],
        'recall': avg_recall[idx],
        'f1_score': avg_f1_score[idx],
    }
    overall_metrics['average_metrics_per_class'].append(class_metric)

# Output Metrics to a JSON File
with open('./results/bert.json', 'w') as json_file:
    json.dump(overall_metrics, json_file, indent=4)

print("\nCross-validation metrics have been written to 'bert.json'")

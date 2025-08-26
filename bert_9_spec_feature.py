import pandas as pd
import numpy as np
import json
import time
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import Counter
import language_tool_python
import os
from sklearn.model_selection import StratifiedKFold

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize language tool for grammar checking
tool = language_tool_python.LanguageTool('en-US')

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load stop words
stop_words = set(stopwords.words('english'))

# Load NRC Emotion Lexicon
def load_nrc_emotion_lexicon():
    nrc_lexicon = {}
    with open('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                # Skip empty lines
                continue
            if line.startswith('#'):
                # Skip comment lines
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                print(f"Skipping line {line_num}: Expected 3 tab-separated values, got {len(parts)}. Line content: {line}")
                continue
            word, emotion, association = parts
            try:
                if int(association) > 0:
                    nrc_lexicon.setdefault(word, []).append(emotion)
            except ValueError:
                print(f"Invalid association value at line {line_num}: {association}")
                continue
    return nrc_lexicon

# Check if the NRC lexicon file exists
if not os.path.isfile('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'):
    print("Please download the NRC Emotion Lexicon file and place it in the current directory.")
    print("Download link: https://saifmohammad.com/WebPages/lexicons.html")
    exit()

nrc_lexicon = load_nrc_emotion_lexicon()
emotional_words = set(nrc_lexicon.keys())

# Define a function to compute features
def compute_features(text):
    features = {}
    
    # Tokenization
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    # Unique word ratio
    unique_words = set(words)
    features['unique_word_ratio'] = len(unique_words) / len(words) if len(words) > 0 else 0
    
    # Number of grammatical errors
    matches = tool.check(text)
    features['grammar_errors'] = len(matches)
    
    # Average word length
    word_lengths = [len(word) for word in words if word.isalpha()]
    features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0
    
    # Average sentence length
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    features['avg_sentence_length'] = np.mean(sentence_lengths) if sentence_lengths else 0
    
    # Number of discourse markers
    discourse_markers = ['however', 'moreover', 'therefore', 'additionally', 'meanwhile', 'consequently']
    features['num_discourse_markers'] = sum(text.lower().count(marker) for marker in discourse_markers)
    
    # POS tagging
    pos_tags = nltk.pos_tag(words)
    pos_counts = Counter(tag for word, tag in pos_tags)
    total_tags = sum(pos_counts.values())
    if total_tags > 0:
        features['noun_ratio'] = (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)) / total_tags
        features['pronoun_ratio'] = (pos_counts.get('PRP', 0) + pos_counts.get('PRP$', 0)) / total_tags
        features['verb_ratio'] = (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)) / total_tags
        features['adverb_ratio'] = (pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + pos_counts.get('RBS', 0)) / total_tags
        features['adjective_ratio'] = (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + pos_counts.get('JJS', 0)) / total_tags
    else:
        features['noun_ratio'] = 0
        features['pronoun_ratio'] = 0
        features['verb_ratio'] = 0
        features['adverb_ratio'] = 0
        features['adjective_ratio'] = 0
    
    # Ratio of punctuation marks to total words
    punctuation_marks = re.findall(r'[^\w\s]', text)
    features['punctuation_ratio'] = len(punctuation_marks) / len(words) if len(words) > 0 else 0
    
    # Ratio of stop words in text
    num_stop_words = sum(1 for word in words if word in stop_words)
    features['stopword_ratio'] = num_stop_words / len(words) if len(words) > 0 else 0
    
    # Ratio of emotional words
    num_emotional_words = sum(1 for word in words if word in emotional_words)
    features['emotional_word_ratio'] = num_emotional_words / len(words) if len(words) > 0 else 0
    
    return features

# Load train and test datasets
train_data = pd.read_csv('./data/train_data.csv')
test_data = pd.read_csv('./data/test_data.csv')

# Combine train and test data for cross-validation
data = pd.concat([train_data, test_data], ignore_index=True)

# Compute features for all data
print("Computing features for all data...")
data_features = data['abstract'].apply(compute_features).apply(pd.Series)
data = pd.concat([data, data_features], axis=1)

# Extract texts, features, and labels
all_texts = data['abstract'].tolist()
all_labels = data['label'].tolist()
all_features_values = data.drop(columns=['abstract', 'label']).values

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a custom Dataset class
class ReviewDataset(Dataset):
    def __init__(self, reviews, features, labels, tokenizer, max_length=128):
        self.reviews = reviews
        self.features = features
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
        features = self.features[idx]
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'features': torch.tensor(features, dtype=torch.float),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Define the model
class CustomBERTModel(nn.Module):
    def __init__(self, bert_model, feature_size, num_labels):
        super(CustomBERTModel, self).__init__()
        self.bert = bert_model
        self.feature_size = feature_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768 + feature_size, num_labels)
        
    def forward(self, ids, mask, token_type_ids, features, labels=None):
        outputs = self.bert(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1]  # [CLS] token representation
        combined_output = torch.cat((pooled_output, features), dim=1)
        combined_output = self.dropout(combined_output)
        logits = self.classifier(combined_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}

# Define training and evaluation functions
def train_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc='Training'):
        optimizer.zero_grad()
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
            features=features,
            labels=labels
        )
        loss = outputs['loss']
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
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids,
                features=features,
                labels=labels
            )
            loss = outputs['loss']
            logits = outputs['logits']
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return total_loss / len(data_loader), pred_labels, true_labels

# Initialize StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store metrics
accuracy_list = []
precision_list = []
recall_list = []
f1_score_list = []
evaluation_time_list = []

# Initialize a list to store metrics for each fold
fold_metrics_list = []

# Ensure the results directory exists
os.makedirs('./results', exist_ok=True)

# Start overall timing
start_time = time.time()

# Class names
class_names = [0, 1]

for fold, (train_idx, val_idx) in enumerate(kfold.split(all_texts, all_labels)):
    print(f"\nFold {fold + 1}/10")
    start_time_fold = time.time()
    
    # Split data
    X_train_texts = [all_texts[i] for i in train_idx]
    X_val_texts = [all_texts[i] for i in val_idx]
    y_train = [all_labels[i] for i in train_idx]
    y_val = [all_labels[i] for i in val_idx]
    X_train_features_values = all_features_values[train_idx]
    X_val_features_values = all_features_values[val_idx]
    
    # Create datasets and data loaders
    train_dataset = ReviewDataset(X_train_texts, X_train_features_values, y_train, tokenizer, max_length=128)
    val_dataset = ReviewDataset(X_val_texts, X_val_features_values, y_val, tokenizer, max_length=128)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize the model
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    feature_size = X_train_features_values.shape[1]
    num_labels = 2
    model = CustomBERTModel(bert_model, feature_size, num_labels)
    model.to(device)
    
    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3  # Assuming 3 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    print("Starting training...")
    for epoch in range(3):  # Number of epochs
        print(f"Epoch {epoch + 1}/3")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"Fold {fold + 1}, Epoch {epoch + 1}, Train loss: {train_loss}")
    
    # Evaluation
    print("Evaluating model on validation data...")
    val_loss, y_pred, y_true = eval_model(model, val_loader)
    print(f"Fold {fold + 1}, Validation loss: {val_loss}")
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0
    )
    
    evaluation_time = time.time() - start_time_fold
    
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
with open('./results/bert_9_spec_feature.json', 'w') as json_file:
    json.dump(overall_metrics, json_file, indent=4)

print("\nCross-validation metrics have been written to 'bert_9_spec_feature.json'")

# Total execution time
total_time = time.time() - start_time
print(f"Total execution time: {total_time} seconds")

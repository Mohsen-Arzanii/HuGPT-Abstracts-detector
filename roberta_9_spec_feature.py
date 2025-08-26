import pandas as pd
import numpy as np
import json
import time
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    RobertaTokenizer,
    RobertaModel,
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
from sklearn.model_selection import StratifiedKFold  # اضافه کردن StratifiedKFold

# بررسی دسترسی به CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"استفاده از دستگاه: {device}")

# مقداردهی اولیه ابزار بررسی گرامر
tool = language_tool_python.LanguageTool('en-US')

# دانلود منابع NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# بارگذاری کلمات توقف
stop_words = set(stopwords.words('english'))

# بارگذاری فرهنگ لغت احساسات NRC
def load_nrc_emotion_lexicon():
    nrc_lexicon = {}
    with open('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            word, emotion, association = parts
            if int(association) > 0:
                if word not in nrc_lexicon:
                    nrc_lexicon[word] = []
                nrc_lexicon[word].append(emotion)
    return nrc_lexicon

# اطمینان از وجود فایل فرهنگ لغت NRC
nrc_lexicon_path = 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
if not os.path.isfile(nrc_lexicon_path):
    print("فایل فرهنگ لغت احساسات NRC یافت نشد. لطفاً آن را دانلود کرده و در دایرکتوری فعلی قرار دهید.")
    print("لینک دانلود: https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm")
    exit()

nrc_lexicon = load_nrc_emotion_lexicon()
emotional_words = set(nrc_lexicon.keys())
print("فرهنگ لغت احساسات NRC با موفقیت بارگذاری شد.")

# تعریف تابع محاسبه ویژگی‌ها
def compute_features(text):
    features = {}
    
    # توکن‌سازی
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    # نسبت کلمات یکتا
    unique_words = set(words)
    features['unique_word_ratio'] = len(unique_words) / len(words) if len(words) > 0 else 0
    
    # تعداد خطاهای گرامری
    matches = tool.check(text)
    features['grammar_errors'] = len(matches)
    
    # میانگین طول کلمات
    word_lengths = [len(word) for word in words]
    features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0
    
    # میانگین طول جملات
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
    features['avg_sentence_length'] = np.mean(sentence_lengths) if sentence_lengths else 0
    
    # تعداد نشانگرهای گفتمانی
    discourse_markers = ['however', 'moreover', 'therefore', 'additionally', 'meanwhile', 'consequently']
    features['num_discourse_markers'] = sum(text.lower().count(marker) for marker in discourse_markers)
    
    # برچسب‌گذاری POS
    pos_tags = nltk.pos_tag(words)
    pos_counts = Counter(tag for word, tag in pos_tags)
    total_tags = sum(pos_counts.values())
    if total_tags > 0:
        features['noun_ratio'] = (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) +
                                  pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)) / total_tags
        features['pronoun_ratio'] = (pos_counts.get('PRP', 0) + pos_counts.get('PRP$', 0)) / total_tags
        features['verb_ratio'] = (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0) +
                                  pos_counts.get('VBN', 0) + pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)) / total_tags
        features['adverb_ratio'] = (pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) +
                                    pos_counts.get('RBS', 0)) / total_tags
        features['adjective_ratio'] = (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) +
                                       pos_counts.get('JJS', 0)) / total_tags
    else:
        features['noun_ratio'] = 0
        features['pronoun_ratio'] = 0
        features['verb_ratio'] = 0
        features['adverb_ratio'] = 0
        features['adjective_ratio'] = 0
    
    # نسبت نشانه‌های نگارشی به کل کلمات
    punctuation_marks = re.findall(r'[^\w\s]', text)
    features['punctuation_ratio'] = len(punctuation_marks) / len(words) if len(words) > 0 else 0
    
    # نسبت کلمات توقف
    num_stop_words = sum(1 for word in words if word in stop_words)
    features['stopword_ratio'] = num_stop_words / len(words) if len(words) > 0 else 0
    
    # نسبت کلمات احساسی
    num_emotional_words = sum(1 for word in words if word in emotional_words)
    features['emotional_word_ratio'] = num_emotional_words / len(words) if len(words) > 0 else 0
    
    return features

# بارگذاری داده‌های آموزشی و تست
train_data = pd.read_csv('./data/train_data.csv')
test_data = pd.read_csv('./data/test_data.csv')

# ترکیب داده‌های آموزشی و تست برای کراس‌ولیدیشن
data = pd.concat([train_data, test_data], ignore_index=True)

# محاسبه ویژگی‌ها برای کل داده‌ها
print("محاسبه ویژگی‌ها برای کل داده‌ها...")
features_df = data['abstract'].apply(compute_features).apply(pd.Series)
data = pd.concat([data, features_df], axis=1)

# استخراج متن‌ها، ویژگی‌ها و برچسب‌ها
texts = data['abstract'].tolist()
labels = data['label'].tolist()
features_final = data.drop(columns=['abstract', 'label']).values

# مقداردهی اولیه توکنایزر
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# تعریف کلاس دیتاست سفارشی
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
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        features = self.features[idx]
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'features': torch.tensor(features, dtype=torch.float),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# تعریف مدل
class CustomRoBERTaModel(nn.Module):
    def __init__(self, feature_size, num_labels):
        super(CustomRoBERTaModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.feature_size = feature_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.roberta.config.hidden_size + feature_size, num_labels)
        
    def forward(self, ids, mask, features, labels=None):
        outputs = self.roberta(
            input_ids=ids,
            attention_mask=mask,
        )
        last_hidden_state = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)
        pooled_output = last_hidden_state[:, 0, :]  # گرفتن [CLS] token
        combined_output = torch.cat((pooled_output, features), dim=1)
        combined_output = self.dropout(combined_output)
        logits = self.classifier(combined_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}

# تعریف توابع آموزش و ارزیابی
def train_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc='آموزش'):
        optimizer.zero_grad()
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            ids=ids,
            mask=mask,
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
        for batch in tqdm(data_loader, desc='ارزیابی'):
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                ids=ids,
                mask=mask,
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

# ایجاد دایرکتوری نتایج در صورت عدم وجود
if not os.path.exists('./results'):
    os.makedirs('./results')

# آماده‌سازی برای ذخیره معیارها
accuracy_list = []
precision_list = []
recall_list = []
f1_score_list = []
evaluation_time_list = []

# تعریف StratifiedKFold
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# لیست برای ذخیره معیارهای هر فولد
fold_metrics_list = []

# نام کلاس‌ها
class_names = sorted(set(labels))

# شروع کراس‌ولیدیشن
for fold, (train_idx, val_idx) in enumerate(kfold.split(texts, labels)):
    print(f'\nFold {fold + 1}/{n_splits}')
    fold_start_time = time.time()
    
    # تقسیم داده‌ها به آموزش و اعتبارسنجی
    X_train_texts = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_train_features = features_final[train_idx]
    
    X_val_texts = [texts[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]
    X_val_features = features_final[val_idx]
    
    # ایجاد دیتاست‌ها و دیتالودرها
    train_dataset = ReviewDataset(X_train_texts, X_train_features, y_train, tokenizer, max_length=128)
    val_dataset = ReviewDataset(X_val_texts, X_val_features, y_val, tokenizer, max_length=128)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # مقداردهی اولیه مدل
    feature_size = X_train_features.shape[1]
    num_labels = len(set(labels))
    model = CustomRoBERTaModel(feature_size, num_labels)
    model.to(device)
    
    # تعریف بهینه‌ساز و scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3  # فرض بر ۳ دوره آموزش
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # آموزش
    print("شروع آموزش...")
    for epoch in range(3):  # تعداد دوره‌ها
        print(f"دوره {epoch + 1}/3")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"Fold {fold + 1}, دوره {epoch + 1}, Loss آموزش: {train_loss}")
    
    # ارزیابی
    print("ارزیابی مدل روی داده‌های اعتبارسنجی...")
    val_loss, y_pred, y_true = eval_model(model, val_loader)
    print(f"Fold {fold + 1}, Loss اعتبارسنجی: {val_loss}")
    
    # محاسبه معیارها
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    evaluation_time = time.time() - fold_start_time
    
    # ذخیره معیارها
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_score_list.append(f1_score)
    evaluation_time_list.append(evaluation_time)
    
    # ذخیره معیارهای مربوط به این فولد
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
    
    # آزادسازی حافظه
    del model
    torch.cuda.empty_cache()

# محاسبه میانگین معیارها
avg_accuracy = np.mean(accuracy_list)
avg_precision = np.mean(precision_list, axis=0)
avg_recall = np.mean(recall_list, axis=0)
avg_f1_score = np.mean(f1_score_list, axis=0)
total_evaluation_time = np.sum(evaluation_time_list)

# تهیه دیکشنری معیارهای کلی
overall_metrics = {
    "time": total_evaluation_time,
    'average_accuracy': avg_accuracy,
    'average_metrics_per_class': [],
    'fold_metrics': fold_metrics_list  # اضافه کردن معیارهای هر فولد
}

for idx, class_name in enumerate(class_names):
    class_metric = {
        'class': str(class_name),
        'precision': avg_precision[idx],
        'recall': avg_recall[idx],
        'f1_score': avg_f1_score[idx],
    }
    overall_metrics['average_metrics_per_class'].append(class_metric)

# خروجی معیارها به فایل JSON
with open('./results/roberta_9_spec_feature.json', 'w') as json_file:
    json.dump(overall_metrics, json_file, indent=4)

print("\nمعیارهای کراس‌ولیدیشن در فایل './results/roberta_9_spec_feature.json' ذخیره شدند.")

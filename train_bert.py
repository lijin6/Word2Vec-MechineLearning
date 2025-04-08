from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import pandas as pd
from datasets import Dataset
import torch

# 1. 准备数据（示例格式）
def load_data(file):
    df = pd.read_csv(file)
    texts = df['text'].tolist()
    labels = df['label'].tolist()  # 0=负面, 1=正面
    return Dataset.from_dict({'text': texts, 'label': labels})

train_data = load_data('data/train.csv')
val_data = load_data('data/val.csv')

# 2. 加载轻量模型
model_name = "bert-base-chinese"  # 或 "hfl/chinese-bert-wwm-ext"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. 数据预处理
def preprocess(examples):
    return tokenizer(examples['text'], truncation=True, max_length=128)

train_data = train_data.map(preprocess, batched=True)
val_data = val_data.map(preprocess, batched=True)

# 4. 训练配置（精简参数）
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,  # 小数据可减少
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True  # 启用混合精度减少显存
)

# 5. 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

trainer.train()
model.save_pretrained("./sentiment_bert")
tokenizer.save_pretrained("./sentiment_bert")
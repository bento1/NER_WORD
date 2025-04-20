from torch.utils.data import DataLoader
from transformers import BertTokenizer,BertTokenizerFast, BertForTokenClassification
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.code.NERDataset import NERDataset
import torch.nn.functional as F
import ast
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# 예시용: Token / Word label 개수



df = pd.read_csv('./dataset/file/ner.csv', encoding='ISO-8859-1')

sentences = df['Sentence'].to_numpy()
tags = df['Tag'].apply(lambda x: ast.literal_eval(x[1:-1])).to_numpy()

X_train, X_tmp, y_train, y_tmp = train_test_split(
    sentences, tags, test_size=1000, random_state=101)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=101)


def get_tags(labels):
    tags = set()
    for l in labels:
        for tag in l:
            tags.add(tag)
    return list(sorted(tags))

tags = get_tags(y_train)
tag2id={t:i for i,t in enumerate(tags)}
id2tag={i:t for i,t in enumerate(tags)}

print(tags)

tokenizer = BertTokenizerFast.from_pretrained("./model/bert-large-multilingual-cased")
model = BertForTokenClassification.from_pretrained("./model/bert-large-multilingual-cased", num_labels=len(tag2id))
model.classifier = nn.Linear(model.config.hidden_size, len(tag2id))
model=model.to(device)
batch_size=16
train_dataset = NERDataset(X_train, y_train, tokenizer, tag2id,512)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = NERDataset(X_val, y_val, tokenizer, tag2id,512)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
from datetime import datetime
path_name=datetime.now().strftime('%Y%m%d%H%M')
writer = SummaryWriter(log_dir=f"./result/{path_name}/runs/ner_experiment")

model.train()
global_step = 0
id2tag[-100]=None
for epoch in range(100):
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        # ✅ TensorBoard에 train loss 기록
        writer.add_scalar("Loss/train", loss.item(), global_step)
        print(f"[Step {global_step}] Train Loss: {loss.item():.4f}")
        # ✅ 매 500 스텝마다 validation loss 계산
        if global_step % 100 == 0 and global_step > 0:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for val_batch in val_loader:
                    val_input_ids = val_batch['input_ids'].to(device)
                    val_attention_mask = val_batch['attention_mask'].to(device)
                    val_labels = val_batch['labels'].to(device)

                    val_outputs = model(
                        input_ids=val_input_ids,
                        attention_mask=val_attention_mask,
                        labels=val_labels
                    )
                    size = val_outputs['logits'].cpu().detach().shape[0] if  F.softmax(val_outputs['logits'], dim=-1).cpu().detach().shape[0]<=batch_size else batch_size
                    prob=F.softmax(val_outputs['logits'].cpu().detach(), dim=-1)
                    original_string = [[id2tag[int(torch.argmax(l,axis=-1).numpy())] for l in prob[b]] for b in range(size)]
                    labletab=[[id2tag[int(l.cpu().detach().numpy())]   for l in val_labels[b] ] for b in range(size) ]
                    token=[tokenizer.decode(val_input_ids[b]) for b in range(size)]
                    print('token : ',token)
                    print('pred : ',original_string)
                    print('ans : ',labletab)
                    val_loss += val_outputs.loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"[Step {global_step}] Train Loss: {avg_val_loss:.4f}")

            writer.add_scalar("Loss/val", avg_val_loss, global_step)
            model.train()

        global_step += 1

    avg_train_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch + 1}] Avg Train Loss: {avg_train_loss:.4f}")

writer.close()
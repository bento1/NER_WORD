import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, sentences, labels, sentence_vectorizer, label_vectorizer, tags, max_len=128):
        self.sentences_ids = sentence_vectorizer.encode(sentences)  # 문장 -> 숫자 인덱스
        self.label_ids = label_vectorizer(labels, tags)             # 라벨 -> 숫자 인덱스 (패딩 포함)
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences_ids)

    def __getitem__(self, idx):
        # 문장과 라벨을 가져옴
        input_ids = self.sentences_ids[idx]
        labels = self.label_ids[idx]
        
        # 패딩을 max_len에 맞춰 추가
        input_ids = input_ids[:self.max_len] + [-100] * (self.max_len - len(input_ids)) if len(input_ids) < self.max_len else input_ids[:self.max_len]
        labels = labels[:self.max_len].tolist() + [-100] * (self.max_len - len(labels)) if len(labels) < self.max_len else labels[:self.max_len]

        # tensor로 변환하여 반환
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }

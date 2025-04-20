from torch.utils.data import Dataset
import torch

class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, tag2id, max_len=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.labels[idx]

        encoding = self.tokenizer([words],
                                return_offsets_mapping=True,
                                is_split_into_words=True,
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_len)

        labels = [-100] * len(encoding['input_ids'])
        word_ids = encoding.word_ids()
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx:
                labels[i] = self.tag2id[tags[word_idx]]
            else:
                labels[i] = self.tag2id[tags[word_idx]]
            previous_word_idx = word_idx

        item = {key: torch.tensor(val) for key, val in encoding.items() if key != 'offset_mapping'}
        item['labels'] = torch.tensor(labels)
        return item
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
        total_count=1
        for t in tags:
            for i, word_idx in enumerate(encoding['offset_mapping'][1:]):
                if word_idx[0]==0 and word_idx[1]==0:
                    continue
            temp_word_token=self.tokenizer([words[word_idx[0]:word_idx[1]+1]])['input_ids'][0][1:]
            count=0
            for j in range(total_count,total_count+len(temp_word_token)):

                if count==0:
                    if tags[i]=='O':
                        labels[j] = self.tag2id[tags[i]]
                        total_count+=1
                    else:
                        tagss=tags[i].split('-')[1]
                        labels[j] = self.tag2id["B-"+tagss]
                        count+=1
                        total_count+=1
                else:
                    if tags[i]=='O':
                        labels[j] = self.tag2id[tags[i]]
                        total_count+=1
                    else:
                        tagss=tags[i].split('-')[1]
                        labels[j] = self.tag2id["I-"+tagss]
                        count+=1
                        total_count+=1
            
        item = {key: torch.tensor(val) for key, val in encoding.items() if key != 'offset_mapping'}
        item['labels'] = torch.tensor(labels)
        return item
import re
from collections import Counter

class SentenceVectorizer:
    def __init__(self, vocab_size=None, lowercase=False):
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.token2idx = {}
        self.idx2token = {}
        self.is_fitted = False

    def tokenize(self, sentence):
        if self.lowercase:
            sentence = sentence.lower()
        # 단순히 공백 기준 분리 (필요하면 더 정교한 tokenizer로 교체 가능)
        return re.findall(r'\w+|\S', sentence)

    def adapt(self, sentences):
        # 모든 문장을 토큰화
        token_list = []
        for sent in sentences:
            tokens = self.tokenize(sent)
            token_list.extend(tokens)

        # 토큰 빈도수 계산
        counter = Counter(token_list)
        most_common = counter.most_common(self.vocab_size)

        # vocab 만들기
        self.token2idx = {token: idx + 2 for idx, (token, _) in enumerate(most_common)}
        self.token2idx['[PAD]'] = 0
        self.token2idx['[UNK]'] = 1
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        self.is_fitted = True

    def get_vocabulary(self):
        if not self.is_fitted:
            raise ValueError("You must call adapt() first!")
        # vocab을 idx 기준으로 정렬해서 리턴
        return [self.idx2token[i] for i in range(len(self.idx2token))]

    def encode(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]

        encoded = []
        for sent in sentences:
            tokens = self.tokenize(sent)
            ids = [self.token2idx.get(token, self.token2idx['[UNK]']) for token in tokens]
            encoded.append(ids)
        return encoded

    def decode(self, ids):
        sentences = []
        for sent_ids in ids:
            tokens = [self.idx2token.get(idx, '[UNK]') for idx in sent_ids]
            sentences.append(' '.join(tokens))
        return sentences
    
import torch
import torch.nn as nn

class LSTMTagger(nn.Module):
    """
    model = LSTMTagger(
    vocab_size=len(word2idx),
    tagset_size=len(tag2id),
    embedding_dim=100,
    hidden_dim=128,
    pad_idx=word2idx["<PAD>"]
    ).to(device)

    batch = next(iter(train_loader))
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    out = model(input_ids=input_ids, labels=labels)
    print(out["loss"], out["logits"].shape)
    """
    
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128, pad_idx=0):
        super(LSTMTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.classifier = nn.Linear(hidden_dim * 2, tagset_size)  # BiLSTM → hidden*2

    def forward(self, input_ids, attention_mask=None, labels=None):
        # input_ids: (batch_size, seq_len)
        embeddings = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeddings)     # (batch_size, seq_len, hidden_dim*2)
        logits = self.classifier(lstm_out)      # (batch_size, seq_len, tagset_size)

        output = {"logits": logits}

        if labels is not None:
            # CrossEntropyLoss는 (N, C), target은 (N,)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_logits = logits.view(-1, logits.shape[-1])
            active_labels = labels.view(-1)
            loss = loss_fct(active_logits, active_labels)
            output["loss"] = loss

        return output
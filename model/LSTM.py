class SimpleTokenizer:
    def __init__(self, sentences, pad_token="<PAD>", unk_token="<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token

        # 단어 집합 만들기
        all_tokens = []
        for sentence in sentences:
            all_tokens.extend(sentence.strip().split())

        vocab = sorted(set(all_tokens))
        self.word2idx = {word: idx + 2 for idx, word in enumerate(vocab)}
        self.word2idx[pad_token] = 0
        self.word2idx[unk_token] = 1
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, sentence):
        return [
            self.word2idx.get(token, self.word2idx[self.unk_token])
            for token in sentence.strip().split()
        ]

    def decode(self, indices):
        return [self.idx2word.get(idx, self.unk_token) for idx in indices]

    def vocab_size(self):
        return len(self.word2idx)

    def pad_sequence(self, sequences, max_len=None):
        if max_len is None:
            max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            seq = seq[:max_len]
            padded.append(seq + [self.word2idx[self.pad_token]] * (max_len - len(seq)))
        return padded
    
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
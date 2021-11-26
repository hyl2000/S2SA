import torch
import torch.nn as nn


class Emo_Classfication(nn.Module):

    def __init__(self, hidden_dim, emotion_vocab, num_layers=1, bidirectional=False, gpu=True, dropout=0.0):
        super(Emo_Classfication, self).__init__()
        self.hidden_dim = hidden_dim
        self.emotion_vocab = emotion_vocab
        self.emotion_size = len(self.emotion_vocab)
        self.gpu = gpu
        self.dropout = dropout

        self.embeddings = nn.Embedding(self.emotion_size, self.hidden_dim)
        self.emotion_encoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional)
        self.hidden_state_encoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional)
        self.state_encoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional)
        # self.lstm2hidden = nn.Linear(2*2*hidden_dim, hidden_dim) 这里因为一开始是双向2层RNN
        self.lstm2hidden = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=self.dropout)
        self.hidden2out = nn.Linear(hidden_dim, self.emotion_size)
        self.loss = nn.CrossEntropyLoss()

    def EmotionClassify(self, context_hidden, emo_context_hidden, emotion_mask):
        self.emotion_encoder.flatten_parameters()
        self.hidden_state_encoder.flatten_parameters()
        self.state_encoder.flatten_parameters()
        _, (s_hidden, s_cell) = self.hidden_state_encoder(context_hidden)
        emb = self.embeddings(emo_context_hidden)
        out, _ = self.emotion_encoder(emb)  # seq_len * batch * hidden
        out = out[:, :, 0:self.hidden_dim]
        out = out.mul(emotion_mask.transpose(0, 1).unsqueeze(2))  # 原mask batch * seq_len
        s_out, (s_hidden, s_cell) = self.state_encoder(out, (s_hidden, s_cell))
        out = self.lstm2hidden(s_hidden)
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        e_hidden = out
        out = self.hidden2out(out[0, :, :].unsqueeze(0))  # batch_size x emotion_size
        out = torch.softmax(out, dim=-1)
        return out, e_hidden

    def forward(self, context_hidden, emotion_context, emotion_mask, next_emotion=None, train=True, batch_first=True):
        # Ture means wgan  --- emotion  得到语境分类结果
        if batch_first:
            result, e_hidden = self.EmotionClassify(context_hidden.transpose(0, 1), emotion_context.transpose(0, 1), emotion_mask)  # (bs, 1)
        else:
            result, e_hidden = self.EmotionClassify(context_hidden, emotion_context, emotion_mask)  # (bs, 1)

        if train:
            gen_emotion_cost = self.loss(result.squeeze(0), next_emotion.squeeze(1))  # batch * c & batch
            return result, gen_emotion_cost, e_hidden
        else:
            return result, e_hidden

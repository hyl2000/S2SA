from EncDecModel import *
from BilinearAttention import *
from EmoClassification import *
import torch.nn.functional as F
from torch.nn import Parameter
import torch


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., weights_dropout=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.weights_dropout = weights_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=False):
        """ Input shape: Time x Batch x Channel
            key_padding_mask: Time x batch
            attn_mask:  tgt_len x src_len
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)
        # k,v: bsz*heads x src_len x dim
        # q: bsz*heads x tgt_len x dim

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # print(query)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(0),
                float(-1e24)
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.transpose(0, 1).unsqueeze(1).unsqueeze(2),
                float(-1e24)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.weights_dropout:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        if not self.weights_dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

            # attn_weights, _ = attn_weights.max(dim=1)
            attn_weights = attn_weights[:, 0, :, :]
            # attn_weights = attn_weights.mean(dim=1)
            attn_weights = attn_weights.transpose(0, 1)
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        x = F.linear(input, weight, bias)
        del weight
        del bias
        return x


class BBCDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, emb_matrix=None, num_layers=4, dropout=0.5):
        super(BBCDecoder, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

        if emb_matrix is None:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        else:
            self.embedding = create_emb_layer(emb_matrix)
        self.embedding_dropout = nn.Dropout(dropout)

        self.src_attn = BilinearAttention(
            query_size=hidden_size, key_size=2 * hidden_size, hidden_size=hidden_size
        )
        self.gru = nn.GRU(2 * hidden_size + embedding_size, hidden_size, bidirectional=False,
                          num_layers=num_layers)

        self.readout = nn.Linear(embedding_size + hidden_size + 2 * hidden_size, hidden_size)

    def forward(self, tgt, state, src_output, src_mask=None):
        embedded = self.embedding(tgt)
        embedded = self.embedding_dropout(embedded)

        src_context, src_attn = self.src_attn(state[:, -1].unsqueeze(1), src_output, src_output,
                                              mask=src_mask.unsqueeze(1))
        src_context = src_context.squeeze(1)
        src_attn = src_attn.squeeze(1)

        gru_input = torch.cat((embedded, src_context), dim=1)
        gru_output, gru_state = self.gru(gru_input.unsqueeze(0), state.transpose(0, 1))
        gru_state = gru_state.transpose(0, 1)

        concat_output = torch.cat((embedded, gru_state[:, -1], src_context), dim=1)

        feature_output = self.readout(concat_output)
        return feature_output, [gru_state], [src_attn]


class S2SA(EncDecModel):
    def __init__(self, embedding_size, hidden_size, vocab2id, id2vocab, max_dec_len=120, beam_width=1, eps=1e-10,
                 emb_matrix=None):
        super(S2SA, self).__init__(vocab2id=vocab2id, max_dec_len=max_dec_len, beam_width=beam_width, eps=eps)

        self.id2vocab = id2vocab
        self.emotion_vocab = {'pad': 0, '认同': 1, '不认同': 2, '开心': 3, '伤心': 4, '惊讶': 5, '好奇': 6, '中立': 7}
        self.hidden_size = hidden_size
        self.beam_width = beam_width

        if emb_matrix is None:
            self.c_embedding = nn.Embedding(len(vocab2id), embedding_size, padding_idx=0)
        else:
            self.c_embedding = create_emb_layer(emb_matrix)
        self.b_embedding = self.c_embedding
        self.c_embedding_dropout = nn.Dropout(0.5)
        self.b_embedding_dropout = nn.Dropout(0.5)

        self.Emo_Class = Emo_Classfication(hidden_size, self.emotion_vocab)

        self.c_enc = nn.GRU(embedding_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.b_enc = nn.GRU(embedding_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.k_enc = nn.GRU(embedding_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.g_enc = nn.GRU(embedding_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        self.attn_linear = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.attention = MultiheadAttention(self.hidden_size, 4)

        self.enc2dec = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.dec = BBCDecoder(embedding_size, hidden_size, len(vocab2id), num_layers=1, dropout=0.5,
                              emb_matrix=emb_matrix)
        self.gen = nn.Linear(self.hidden_size, len(vocab2id))

    def encode(self, data):
        c_mask = data['context'].ne(0).detach()
        b_mask = data['reverse'].ne(0).detach()

        c_words = self.c_embedding_dropout(self.c_embedding(data['context']))
        b_words = self.c_embedding_dropout(self.c_embedding(data['reverse']))

        c_lengths = c_mask.sum(dim=1).detach()
        b_lengths = b_mask.sum(dim=1).detach()
        c_enc_output, c_state = gru_forward(self.c_enc, c_words, c_lengths)
        b_enc_output, b_state = gru_forward(self.b_enc, b_words, b_lengths)

        KL_Loss = F.kl_div(c_state.softmax(dim=-1).log(), b_state.softmax(dim=-1), reduction='sum')

        g_mask = data['goal'].ne(0).detach()
        g_words = self.c_embedding_dropout(self.c_embedding(data['goal']))
        g_lengths = g_mask.sum(dim=1).detach()
        g_enc_output, g_state = gru_forward(self.g_enc, g_words, g_lengths)

        knowledge = self.c_embedding_dropout(self.c_embedding(data['knowledge']))
        batch_size, knowledge_num, _ = knowledge.size()
        knowledge = knowledge.transpose(0, 2)
        hidden_knowledge = None
        knowledge_output = None
        for i in range(knowledge_num):
            knowledge_n = knowledge[:, i, :]
            knowledge_output_t, hidden_knowledge_t = gru_forward(self.k_enc, knowledge_n, b_lengths)
            if i == 0:
                hidden_knowledge = hidden_knowledge_t
                knowledge_output = knowledge_output_t.unsqueeze(0)
            else:
                hidden_knowledge = torch.cat((hidden_knowledge, hidden_knowledge_t), dim=0)
                knowledge_output = torch.cat((knowledge_output, knowledge_output_t.unsqueeze(0)), dim=0)

        emotion = data['emotion']
        emotion_mask = data['emotion'].ne(0).detach()
        next_emo = data['next_emotion']
        emotion_class_hidden = torch.cat((c_state, hidden_knowledge), 0)
        e, Loss_e, e_hidden = self.Emo_Class(emotion_class_hidden, emotion, emotion_mask, next_emotion=next_emo, train=True)
        pre_e = e.argmax(dim=-1)  # batch_size * 1
        x = pre_e.cpu().squeeze(0).numpy()
        y = next_emo.cpu().squeeze(0).numpy()
        count = 0
        for i in range(batch_size):
            if x[i] == y[i]:
                count += 1

        hidden_sum = torch.cat((c_state, e_hidden, g_state), dim=2)
        att_hidden = self.attn_linear(hidden_sum)
        knowledge_mask = data['knowledge'].ne(0).detach()
        knowledge_mask = torch.logical_not(knowledge_mask.transpose(0, 1).bool())

        attn, attn_weights = self.attention(att_hidden, hidden_knowledge, hidden_knowledge, key_padding_mask=knowledge_mask, need_weights=True)
        # 1 * batch * hidden & 1 * batch * knowledge_num
        sel_knowledge_idx = attn_weights.argmax(dim=-1).squeeze(0)  # batch_size
        sel_knowledge = None  # 1 * batch_size * hidden
        sel_knowledge_input = None  # batch * len
        sel_knowledge_encode = None  # batch * len * hidden
        for i in range(batch_size):
            if i == 0:
                sel_knowledge = hidden_knowledge[sel_knowledge_idx[i], i, :].unsqueeze(0).unsqueeze(1)
                sel_knowledge_input = knowledge[:, sel_knowledge_idx[i], i].unsqueeze(0)
                sel_knowledge_encode = knowledge_output[sel_knowledge_idx[i], :, i, :].unsqueeze(0)
            else:
                sel_knowledge = torch.cat((sel_knowledge, hidden_knowledge[sel_knowledge_idx[i], i, :].unsqueeze(0).unsqueeze(1)), dim=1)
                sel_knowledge_input = torch.cat((sel_knowledge_input, knowledge[:, sel_knowledge_idx[i], i].unsqueeze(0)), dim=0)
                sel_knowledge_encode = torch.cat((sel_knowledge_encode, knowledge_output[sel_knowledge_idx[i], :, i, :].unsqueeze(0)), dim=0)

        decoder_hidden = torch.cat((hidden_sum, sel_knowledge), dim=2)

        return (c_enc_output, decoder_hidden), KL_Loss, Loss_e

    def init_decoder_states(self, data, encode_output):
        c_state = encode_output[1]
        batch_size = encode_output[0].size(0)

        return self.enc2dec(c_state.contiguous().view(batch_size, -1)).view(batch_size, 1, -1)

    def decode(self, data, previous_word, encode_outputs, previous_deocde_outputs):
        c_mask = data['context'].ne(0)
        feature_output, [gru_state], [src_attn] = self.dec(previous_word, previous_deocde_outputs['state'],
                                                           encode_outputs[0], src_mask=c_mask)
        return {'state': gru_state, 'feature': feature_output}

    def generate(self, data, encode_outputs, decode_outputs, softmax=True):
        return self.gen(decode_outputs['feature'])

    def to_word(self, data, gen_output, k=5, sampling=False):
        if not sampling:
            return topk(gen_output, k=k)
        else:
            return randomk(gen_output, k=k)

    def to_sentence(self, data, batch_indices):
        return to_sentence(batch_indices, self.id2vocab)

    def mle_train(self, data):
        encode_output, init_decoder_state, all_decode_output, all_gen_output, KL_loss, Emotion_loss = \
            decode_to_end(self, data, self.vocab2id, tgt=data['response'])
        gen_output = torch.cat([p.unsqueeze(1) for p in all_gen_output], dim=1)
        loss = F.cross_entropy(gen_output.view(-1, gen_output.size(-1)), data['response'].view(-1), ignore_index=0)
        loss = loss.unsqueeze(0) * 0.7 + Emotion_loss * 0.2 + KL_loss * 0.1
        return loss.unsqueeze(0)

    def forward(self, data, method='mle_train'):
        if method == 'mle_train':
            return self.mle_train(data)
        elif method == 'test':
            if self.beam_width == 1:
                return self.greedy(data)
            else:
                return self.beam(data)

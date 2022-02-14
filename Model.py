from EncDecModel import *
from BilinearAttention import *
from EmoClassification import *
from RGCN import *
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, Adafactor, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.models.t5.configuration_t5 import T5Config
import copy
import torch


class Config(nn.Module):
    def __init__(self):
        super(Config, self).__init__()
        self.vocab_size = None
        self.hidden_size = None
        self.pad_token_id = None
        self.max_entity_embeddings = None
        self.max_position_embeddings = None
        self.max_triple_embeddings = None
        self.layer_norm_eps = None
        self.hidden_dropout_prob = None


class KnowledgeEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(KnowledgeEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.entity_embeddings = nn.Embedding(config.max_entity_embeddings, config.hidden_size)
        self.triple_embeddings = nn.Embedding(config.max_triple_embeddings, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, entity_ids, triple_ids, position_ids):
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        inputs_embeds = self.word_embeddings(input_ids)
        entity_embeddings = self.entity_embeddings(entity_ids)
        triple_embeddings = self.triple_embeddings(triple_ids)
        position_embeddings = self.position_embeddings(triple_ids)

        embeddings = inputs_embeds + entity_embeddings + triple_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class S2SA(EncDecModel):
    def __init__(self, embedding_size, hidden_size, vocab2id, id2vocab, entity2id, relation2id, pretrain_model=None,
                 max_dec_len=120, beam_width=1, eps=1e-12, emb_matrix=None):
        super(S2SA, self).__init__(vocab2id=vocab2id, max_dec_len=max_dec_len, beam_width=beam_width, eps=eps)

        self.method = "mle_train"
        self.id2vocab = id2vocab
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.emotion_vocab = {'pad': 0, '认同': 1, '不认同': 2, '开心': 3, '伤心': 4, '惊讶': 5, '好奇': 6, '中立': 7}
        self.hidden_size = hidden_size
        self.beam_width = beam_width

        self.Emo_Class = Emo_Classfication(hidden_size, self.emotion_vocab, num_layers=1, bidirectional=True)

        # config = T5Config()
        config = pretrain_model.config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        knowledge_config = Config()
        knowledge_config.vocab_size = config.vocab_size
        knowledge_config.hidden_size = config.d_model
        knowledge_config.pad_token_id = 0
        knowledge_config.max_entity_embeddings = len(entity2id)
        knowledge_config.max_position_embeddings = 1024
        knowledge_config.max_triple_embeddings = len(relation2id)
        knowledge_config.layer_norm_eps = eps
        knowledge_config.hidden_dropout_prob = 0.1
        self.knowledge_embedding = KnowledgeEmbeddings(knowledge_config)
        self.c_enc = T5Stack(encoder_config, self.shared)
        self.b_enc = T5Stack(encoder_config, self.shared)
        self.g_enc = T5Stack(encoder_config, self.shared)
        self.c_enc.load_state_dict(pretrain_model.get_encoder().state_dict())
        self.b_enc.load_state_dict(pretrain_model.get_encoder().state_dict())
        self.g_enc.load_state_dict(pretrain_model.get_encoder().state_dict())
        self.k_enc = T5Stack(encoder_config, self.knowledge_embedding)

        self.knowledge_linear = nn.Linear(100, self.hidden_size)

        self.enc2dec = nn.Linear(self.hidden_size, self.hidden_size)
        self.dec = pretrain_model.get_decoder()
        self.gen = nn.Linear(self.hidden_size, config.vocab_size)

    def encode(self, data):
        c_mask = data['context'].ne(0).detach()
        b_mask = data['reverse'].ne(0).detach()

        batch_size, _ = data['reverse'].size()
        c_output = self.c_enc(input_ids=data['context'], attention_mask=c_mask)
        c_enc_output = c_output.last_hidden_state  # last hidden: batch_size * length * hidden_size

        g_mask = data['goal'].ne(0).detach()
        g_output = self.g_enc(input_ids=data['goal'], attention_mask=g_mask)
        g_state = g_output.last_hidden_state  # last hidden: batch_size * length * hidden_size

        knowledge = data['knowledge']
        entity_ids = data['entity_ids']
        triple_ids = data['triple_ids']
        position_ids = data['position_ids']
        knowledge_mask = torch.logical_not(knowledge.ne(0).detach())
        knowlegde_emb = self.knowledge_embedding(knowledge, entity_ids, triple_ids, position_ids)
        # k_output = self.k_enc(input_ids=knowledge, attention_mask=knowledge_mask, inputs_embeds=knowlegde_emb)
        k_output = self.k_enc(attention_mask=knowledge_mask, inputs_embeds=knowlegde_emb)
        k_state = k_output.last_hidden_state

        emotion = data['emotion']
        emotion_mask = data['emotion'].ne(0).detach()
        next_emo = data['next_emotion']
        emotion_class_hidden = torch.cat((c_enc_output, g_state, k_state), 1)
        if self.method == 'mle_train':
            e, Loss_e, e_hidden = self.Emo_Class(emotion_class_hidden, emotion, emotion_mask, next_emotion=next_emo, train=True)
        else:
            e, e_hidden = self.Emo_Class(emotion_class_hidden, emotion, emotion_mask, next_emotion=next_emo, train=False)
        pre_e = e.argmax(dim=-1)  # batch_size * 1
        x = pre_e.cpu().squeeze(0).numpy()
        y = next_emo.cpu().squeeze(0).numpy()
        count = 0
        for i in range(batch_size):
            if x[i] == y[i]:
                count += 1

        hidden_sum = torch.cat((c_enc_output, e_hidden.transpose(0, 1), g_state), dim=1)

        c_enc_output = torch.cat((c_enc_output[:, :, 0:self.hidden_size], k_state), dim=1)
        decoder_hidden = hidden_sum[:, 0, :].unsqueeze(1)

        if self.method == 'mle_train':
            return (c_enc_output, decoder_hidden), Loss_e, count, knowledge_mask
        else:
            return (c_enc_output, decoder_hidden), count, knowledge_mask

    def init_decoder_states(self, data, encode_output):
        c_state = encode_output[1][:, 0, :]
        batch_size = encode_output[0].size(0)

        return self.enc2dec(c_state.contiguous().view(batch_size, -1)).view(batch_size, 1, -1)

    def decode(self, data, previous_word, encode_outputs, previous_deocde_outputs, knowledge_mask):
        c_mask = torch.cat((data['context'].ne(0), knowledge_mask.transpose(0, 1)), dim=1)
        decoder_outputs = self.dec(
            input_ids=previous_word.unsqueeze(1),
            # attention_mask=c_mask,
            encoder_hidden_states=encode_outputs[0],
            encoder_attention_mask=c_mask
        )

        return {'state': decoder_outputs[0], 'feature': decoder_outputs.last_hidden_state}

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
        encode_output, init_decoder_state, all_decode_output, all_gen_output, Emotion_loss, count =\
            decode_to_end(self, data, self.vocab2id, tgt=data['response'])
        gen_output = torch.cat([p.unsqueeze(1) for p in all_gen_output], dim=1)
        loss = F.cross_entropy(gen_output.view(-1, gen_output.size(-1)), data['response'].view(-1), ignore_index=0)
        loss = loss.unsqueeze(0) * 0.7 + Emotion_loss * 0.3
        return loss.unsqueeze(0), count

    def forward(self, data, method='mle_train'):
        self.method = method
        if method == 'mle_train':
            return self.mle_train(data)
        elif method == 'test':
            if self.beam_width == 1:
                return self.greedy(data)
            else:
                return self.beam(data)

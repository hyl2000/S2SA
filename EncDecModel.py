from Generations import *
import torch.nn as nn


class EncDecModel(nn.Module):
    def __init__(self, max_dec_len=120, beam_width=1, eps=1e-10):
        super(EncDecModel, self).__init__()
        self.eps = eps
        self.beam_width = beam_width
        self.max_dec_len = max_dec_len

    def encode(self, data):
        raise NotImplementedError

    def init_decoder_states(self, data, encode_output):
        return None

    def decode(self, data, previous_word, encode_outputs, previous_deocde_outputs, knowledge_mask):
        raise NotImplementedError

    def generate(self, data, encode_outputs, decode_outputs, softmax=False):
        raise NotImplementedError

    def to_word(self, data, gen_outputs, k=5, sampling=False):
        raise NotImplementedError

    def generation_to_decoder_input(self, data, indices):
        return indices

    def loss(self,data, encode_output,decode_outputs, gen_outputs, reduction='mean'):
        raise NotImplementedError

    def to_sentence(self, data, batch_indice, tokenizer):
        raise NotImplementedError

    def sample(self, data):
        raise NotImplementedError

    def greedy(self, data):
        return greedy(self,data,self.tokenizer, self.max_dec_len)

    def beam(self, data):
        return beam(self, data, self.tokenizer, self.max_dec_len, self.beam_width)
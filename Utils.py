from Constants import *
from torch.nn.init import *
import torch.nn.functional as F
import pickle
# import bcolz
import torch.nn as nn
import numpy as np
import random
import time
import codecs
from tqdm import tqdm
from torch.distributions.categorical import *
from torch_geometric.data import Data
from torch_scatter import scatter_add


def get_ms():
    return time.time() * 1000


def init_seed(seed=None):
    if seed is None:
        seed = int(get_ms() // 1000)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def importance_sampling(prob, topk):
    m = Categorical(logits=prob)
    indices = m.sample((topk,)).transpose(0, 1)  # batch, topk

    values = prob.gather(1, indices)
    return values, indices


def decode_to_end(model, data, vocab2id, max_target_length=None, schedule_rate=1, softmax=False, encode_outputs=None,
                  init_decoder_states=None, tgt=None):
    # if tgt is None:
    #     tgt = data['output']
    batch_size = len(data['id'])
    if max_target_length is None:
        max_target_length = tgt.size(1)

    if encode_outputs is None:
        encode_outputs, KL_loss, Emotion_loss, count = model.encode(data)
    if init_decoder_states is None:
        init_decoder_states = model.init_decoder_states(data, encode_outputs)

    decoder_input = new_tensor([vocab2id[BOS_WORD]] * batch_size, requires_grad=False)

    prob = torch.ones((batch_size,)) * schedule_rate
    if torch.cuda.is_available():
        prob = prob.cuda()

    all_gen_outputs = list()
    all_decode_outputs = [dict({'state': init_decoder_states})]

    for t in range(max_target_length):
        # decoder_outputs, decoder_states,...
        decode_outputs = model.decode(
            data, decoder_input, encode_outputs, all_decode_outputs[-1]
        )

        output = model.generate(data, encode_outputs, decode_outputs, softmax=softmax)

        all_gen_outputs.append(output)
        all_decode_outputs.append(decode_outputs)

        if schedule_rate >= 1:
            decoder_input = tgt[:, t]
        elif schedule_rate <= 0:
            probs, ids = model.to_word(data, output, 1)
            decoder_input = model.generation_to_decoder_input(data, ids[:, 0])
        else:
            probs, ids = model.to_word(data, output, 1)
            indices = model.generation_to_decoder_input(data, ids[:, 0])

            draws = torch.bernoulli(prob).long()
            decoder_input = tgt[:, t] * draws + indices * (1 - draws)

    # all_gen_outputs = torch.cat(all_gen_outputs, dim=0).transpose(0, 1).contiguous()

    return encode_outputs, init_decoder_states, all_decode_outputs, all_gen_outputs, KL_loss, Emotion_loss, count


def randomk(gen_output, k=5, PAD=None, BOS=None, UNK=None):
    if PAD is not None:
        gen_output[:, PAD] = -float('inf')
    if BOS is not None:
        gen_output[:, BOS] = -float('inf')
    if UNK is not None:
        gen_output[:, UNK] = -float('inf')
    values, indices = importance_sampling(gen_output, k)
    # words=[[tgt_id2vocab[id.item()] for id in one] for one in indices]
    return values, indices


def topk(gen_output, k=5, PAD=None, BOS=None, UNK=None):
    if PAD is not None:
        gen_output[:, PAD] = 0
    if BOS is not None:
        gen_output[:, BOS] = 0
    if UNK is not None:
        gen_output[:, UNK] = 0
    if k > 1:
        values, indices = torch.topk(gen_output, k, dim=1, largest=True,
                                     sorted=True, out=None)
    else:
        values, indices = torch.max(gen_output, dim=1, keepdim=True)
    return values, indices


def copy_topk(gen_output, vocab_map, vocab_overlap, k=5, PAD=None, BOS=None, UNK=None):
    vocab = gen_output[:, :vocab_map.size(-1)]
    dy_vocab = gen_output[:, vocab_map.size(-1):]

    vocab = vocab + torch.bmm(dy_vocab.unsqueeze(1), vocab_map).squeeze(1)
    dy_vocab = dy_vocab * vocab_overlap

    gen_output = torch.cat([vocab, dy_vocab], dim=-1)
    return topk(gen_output, k, PAD=PAD, BOS=BOS, UNK=UNK)


def remove_duplicate_once(sents, n=3):
    changed = False
    for b in range(len(sents)):
        sent = sents[b]
        if len(sent) <= n:
            continue

        for i in range(len(sent) - n):
            index = len(sent) - i - n
            if all(elem in sent[:index] for elem in sent[index:]):
                sents[b] = sent[:index]
                changed = True
                break
    return changed


def remove_duplicate(sents, n=3):
    changed = remove_duplicate_once(sents, n)
    while changed:
        changed = remove_duplicate_once(sents, n)


def to_sentence(batch_indices, id2vocab):
    batch_size = len(batch_indices)
    summ = list()
    for i in range(batch_size):
        indexes = batch_indices[i]
        text_summ2 = []
        for index in indexes:
            index = index.item()
            w = id2vocab[index]
            if w == BOS_WORD or w == PAD_WORD:
                continue
            if w == EOS_WORD:
                break
            text_summ2.extend(w)
        if len(text_summ2) == 0:
            text_summ2.extend(UNK_WORD)
        summ.append(text_summ2)
    return summ


def to_copy_sentence(data, batch_indices, id2vocab, dyn_id2vocab_map):
    ids = data['id']
    batch_size = len(batch_indices)
    summ = list()
    for i in range(batch_size):
        indexes = batch_indices[i]
        text_summ2 = []
        dyn_id2vocab = dyn_id2vocab_map[ids[i].item()]
        for index in indexes:
            index = index.item()
            if index < len(id2vocab):
                w = id2vocab[index]
            elif index - len(id2vocab) in dyn_id2vocab:
                w = dyn_id2vocab[index - len(id2vocab)]
            else:
                w = PAD_WORD

            if w == BOS_WORD or w == PAD_WORD:
                continue

            if w == EOS_WORD:
                break

            text_summ2.append(w)

        if len(text_summ2) == 0:
            text_summ2.append(UNK_WORD)

        summ.append(text_summ2)
    return summ


def create_emb_layer(emb_matrix, non_trainable=True):
    num_embeddings, embedding_dim = emb_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
    emb_layer.load_state_dict({'weight': emb_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer


'''
def load_embeddings(emb_text_filepath, id2vocab, emb_dim):
    vectors = bcolz.open(emb_text_filepath + '.dat')[:]
    words = pickle.load(open(emb_text_filepath + '.words.pkl', 'rb'))
    word2idx = pickle.load(open(emb_text_filepath + '.ids.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    matrix_len = len(id2vocab)
    emb_matrix = torch.zeros((matrix_len, emb_dim))
    words_found = 0

    for i in range(len(id2vocab)):
        word = id2vocab[i]
        try:
            emb_matrix[i] = torch.Tensor(glove[word])
            words_found += 1
        except KeyError:
            emb_matrix[i] = torch.Tensor(np.random.normal(scale=0.6, size=(emb_dim,)))
    return emb_matrix


def prepare_embeddings(emb_text_filepath):
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=emb_text_filepath + '.temp', mode='w')

    with open(emb_text_filepath, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((idx, -1)), rootdir=emb_text_filepath + '.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(emb_text_filepath + '.words.pkl', 'wb'))
    pickle.dump(word2idx, open(emb_text_filepath + '.ids.pkl', 'wb'))
'''


def new_tensor(array, requires_grad=False):
    tensor = torch.tensor(array, requires_grad=requires_grad)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


# def hotfix_pack_padded_sequence(input, lengths, batch_first=True):
#     lengths = torch.as_tensor(lengths, dtype=torch.int64)
#     lengths = lengths.cpu()
#     return PackedSequence(torch._C._VariableFunctions._pack_padded_sequence(input, lengths, batch_first))

def gru_forward(gru, input, lengths, state=None, batch_first=True):
    gru.flatten_parameters()
    input_lengths, perm = torch.sort(lengths, descending=True)

    input = input[perm]
    if state is not None:
        state = state[perm].transpose(0, 1).contiguous()

    total_length = input.size(1)
    if not batch_first:
        input = input.transpose(0, 1)  # B x L x N -> L x B x N
    packed = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths.cpu(), batch_first)
    # packed = hotfix_pack_padded_sequence(embedded, input_lengths, batch_first)
    # self.gru.flatten_parameters()
    outputs, state = gru(packed, state)  # -> L x B x N * n_directions, 1, B, N
    outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=batch_first,
                                                                     total_length=total_length)  # unpack (back to padded)

    _, perm = torch.sort(perm, descending=False)
    if not batch_first:
        outputs = outputs.transpose(0, 1)
    outputs = outputs[perm]
    state = state.transpose(0, 1)[perm]

    return outputs, state


def build_map(b_map, max=None):
    batch_size, b_len = b_map.size()
    if max is None:
        max = b_map.max() + 1
    b_map_ = torch.zeros(batch_size, b_len, max)
    if torch.cuda.is_available():
        b_map_ = b_map_.cuda()
    b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
    # b_map_[:, :, 0] = 0.
    b_map_.requires_grad = False
    return b_map_


def build_vocab(words, max=100000):
    dyn_vocab2id = dict({PAD_WORD: 0})
    dyn_id2vocab = dict({0: PAD_WORD})
    for w in words:
        if w not in dyn_vocab2id and len(dyn_id2vocab) < max:
            dyn_vocab2id[w] = len(dyn_vocab2id)
            dyn_id2vocab[len(dyn_id2vocab)] = w
    return dyn_vocab2id, dyn_id2vocab


def merge1D(sequences, max_len=None, pad_value=None):
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths) if max_len is None else max_len
    if pad_value is None:
        padded_seqs = torch.zeros(len(sequences), max_len, requires_grad=False).type_as(sequences[0])
    else:
        padded_seqs = torch.full((len(sequences), max_len), pad_value, requires_grad=False).type_as(sequences[0])

    for i, seq in enumerate(sequences):
        end = min(lengths[i], max_len)
        padded_seqs[i, :end] = seq[:end]
    return padded_seqs


def get_data(i, data):
    ones = dict()
    for key, value in data.items():
        if value is not None:
            if isinstance(value, torch.Tensor):
                ones[key] = value[i].unsqueeze(0)
            elif isinstance(value, dict):
                ones[key] = value[data['id'][i].item()]
            else:
                ones[key] = [value[i]]
        else:
            ones[key] = None
    return ones


def concat_data(datalist):
    data = dict()

    size = len(datalist)

    for key in datalist[0]:
        value = datalist[0][key]
        if value is not None:
            if isinstance(value, torch.Tensor):
                data[key] = torch.cat([datalist[i][key] for i in range(size)], dim=0)
            elif isinstance(value, dict):
                data[key] = dict()
                for i in range(size):
                    data[key][datalist[i]['id'].item()] = datalist[i][key]
            else:
                data[key] = [datalist[i][key] for i in range(size)]
        else:
            data[key] = None
    return data


def load_vocab(vocab_file, entities_file, relations_file, t=0):
    thisvocab2id = dict({PAD_WORD: 0, BOS_WORD: 1, UNK_WORD: 2, EOS_WORD: 3, SEP_WORD: 4, CLS_WORD: 5, MASK_WORD: 6})
    thisid2vocab = dict({0: PAD_WORD, 1: BOS_WORD, 2: UNK_WORD, 3: EOS_WORD, 4: SEP_WORD, 5: CLS_WORD, 6: MASK_WORD})
    entity2id = dict()
    relation2id = dict()

    with codecs.open(vocab_file, encoding='utf-8') as f:
        for line in f:
            try:
                name = line.strip('\n').strip('\r').split('\t')
            except Exception:
                continue
            id = len(thisvocab2id)
            thisvocab2id[name[0]] = id
            thisid2vocab[id] = name

    with codecs.open(entities_file, encoding='utf-8') as f:
        for line in f:
            try:
                name = line.strip('\n').strip('\r').split('\t')
            except Exception:
                continue
            id = len(entity2id)
            entity2id[name[0]] = id

    with codecs.open(relations_file, encoding='utf-8') as f:
        for line in f:
            try:
                name = line.strip('\n').strip('\r').split('\t')
            except Exception:
                continue
            id = len(relation2id)
            relation2id[name[0]] = id

    print('vocab item size: ', len(thisvocab2id))
    print('entity item size: ', len(entity2id))
    print('relation item size: ', len(relation2id))

    return thisvocab2id, thisid2vocab, entity2id, relation2id


def load_embedding(src_vocab2id, file):
    model = dict()
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            model[word] = torch.tensor([float(val) for val in splitLine[1:]])
    matrix = torch.zeros((len(src_vocab2id), 100))
    xavier_uniform_(matrix)
    for word in model:
        if word in src_vocab2id:
            matrix[src_vocab2id[word]] = model[word]
    return matrix


def read_triplets(file_path, entity2id, relation2id):
    triplets = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    return np.array(triplets)


def sample_edge_uniform(n_triples, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triples)
    return np.random.choice(all_edges, sample_size, replace=False)


def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.choice(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels


def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    '''
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    '''
    one_hot = F.one_hot(edge_type, num_classes=2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim=0, dim_size=num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm


def generate_sampled_graph_and_labels(triplets, sample_size, split_size, num_entity, num_rels, negative_rate):
    """
        Get training graph and signals
        First perform edge neighborhood sampling on graph, then perform negative
        sampling to generate negative samples
    """

    edges = sample_edge_uniform(len(triplets), sample_size)

    # Select sampled edges
    edges = np.array(triplets)[edges]
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # Negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_entity), negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)

    src = torch.tensor(src[graph_split_ids], dtype=torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype=torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype=torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(uniq_entity).long()
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    data.samples = torch.from_numpy(samples).long()
    data.labels = torch.from_numpy(labels).long()
    if torch.cuda.is_available():
        data = data.cuda()

    return data


def build_test_graph(num_nodes, num_rels, triplets):
    src, rel, dst = triplets.transpose()

    src = torch.from_numpy(src.astype(np.int64))
    rel = torch.from_numpy(rel.astype(np.int64))
    dst = torch.from_numpy(dst.astype(np.int64))

    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index=edge_index)
    data.entity = torch.from_numpy(np.arange(num_nodes))
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, num_nodes, num_rels)

    return data


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def calc_mrr(embedding, w, test_triplets, all_triplets, hits=[]):
    with torch.no_grad():

        num_entity = len(embedding)

        ranks_s = []
        ranks_o = []

        head_relation_triplets = all_triplets[:, :2]
        tail_relation_triplets = torch.stack((all_triplets[:, 2], all_triplets[:, 1])).transpose(0, 1)

        for test_triplet in tqdm(test_triplets):
            # Perturb object
            subject = test_triplet[0]
            relation = test_triplet[1]
            object_ = test_triplet[2]

            subject_relation = test_triplet[:2]
            delete_index = torch.sum(head_relation_triplets == subject_relation, dim=1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, object_.view(-1)))

            emb_ar = embedding[subject] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim=0)
            score = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_s.append(sort_and_rank(score, target))

            # Perturb subject
            object_ = test_triplet[2]
            relation = test_triplet[1]
            subject = test_triplet[0]

            object_relation = torch.tensor([object_, relation])
            delete_index = torch.sum(tail_relation_triplets == object_relation, dim=1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            delete_entity_index = all_triplets[delete_index, 0].view(-1).numpy()
            perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
            perturb_entity_index = torch.from_numpy(perturb_entity_index)
            perturb_entity_index = torch.cat((perturb_entity_index, subject.view(-1)))

            emb_ar = embedding[object_] * w[relation]
            emb_ar = emb_ar.view(-1, 1, 1)

            emb_c = embedding[perturb_entity_index]
            emb_c = emb_c.transpose(0, 1).unsqueeze(1)

            out_prod = torch.bmm(emb_ar, emb_c)
            score = torch.sum(out_prod, dim=0)
            score = torch.sigmoid(score)

            target = torch.tensor(len(perturb_entity_index) - 1)
            ranks_o.append(sort_and_rank(score, target))

        ranks_s = torch.cat(ranks_s)
        ranks_o = torch.cat(ranks_o)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1  # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))

    return mrr.item()

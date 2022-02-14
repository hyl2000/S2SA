from torch.utils.data import Dataset
from Utils import *
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, Adafactor, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import json
import re


def takeSecond(elem):
    return elem[0]


def pad_knowledge(knowledge):
    for i in range(len(knowledge)):
        if len(knowledge[i]) > 50:
            knowledge[i] = knowledge[i][0:50]
        else:
            knowledge[i] = knowledge[i] + (50 - len(knowledge[i])) * [0]
    return knowledge


def linearize_v2(tokenizer, entity):
    string = tokenizer.encode('[ENT] {}'.format(entity[0]), add_special_tokens=False)  # entity[0]代表A实体
    triple_id = [1] * len(string)

    added = set()
    for rel in entity[1]:  # 所有开始是A实体的关系和实体B
        # if self.forbid_duplicate_relation and rel[0] in added:
        if rel[0] in added:
            pass
        else:
            words = tokenizer.encode('[PRED] {} [SUB] {} [TRIPLE]'.format(rel[0], rel[1]), add_special_tokens=False)
            string += words
            triple_id += [triple_id[-1] + 1] * len(words)
            added.add(rel[0])

        if len(added) >= 1024:
            break

    return string, triple_id


class Dataset(Dataset):
    def __init__(self, path, vocab2id, entity2id, relation2id, batch_size, model_name, valid_path=None, max_length=200, n=1E10):
        super(Dataset, self).__init__()

        self.emotion_vocab = {'pad': 0, '认同': 1, '不认同': 2, '开心': 3, '伤心': 4, '惊讶': 5, '好奇': 6, '中立': 7}
        self.pad_idx = 0
        self.max_enc_len = 1024

        self.max_length = max_length
        self.path = path
        self.valid_path = valid_path
        self.batch_size = batch_size

        self.response = []
        self.context = []

        self.vocab2id = vocab2id
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.model_name = model_name
        # self.knowledge_data = None
        self.GCN_train_sample = None
        self.GCN_valid_sample = None
        self.build_graph_data = None
        self.all_triplets = None
        self.n = n

        self.sample_tensor = []
        self.samples = []
        self.len = 0
        self.load()

    def load(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        with open(self.path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                session = json.loads(line.strip(), encoding="utf-8")
                self.samples.append(session)
        # all_knowledge = []
        for idx in range(len(self.samples)):
            sample = self.samples[idx]
            id_tensor = torch.tensor([idx]).long()
            for i in range(len(sample["history"])):
                if sample["history"][i].startswith('['):
                    sample["history"][i] = sample["history"][i][4:]

            for i in range(len(sample["reverse"])):
                if sample["reverse"][i].startswith('['):
                    sample["reverse"][i] = sample["reverse"][i][4:]

            if sample["response"].startswith('['):
                sample["response"] = sample["response"][4:]

            contexts = (' ' + SEP_WORD + ' ').join(sample["history"][2:])
            contexts = ''.join(contexts.split(' '))

            while len(contexts) < 2:
                contexts = '<nan>' + contexts
            # self.context.append(' '.join(contexts))
            self.context.append(contexts)
            contexts = tokenizer.encode(contexts)
            context_tensor = torch.tensor(contexts[-self.max_length:])

            reverse = (' ' + SEP_WORD + ' ').join(sample["reverse"])
            reverse = ''.join(reverse.split(' '))

            while len(reverse) < 2:
                reverse = '<nan>' + reverse
            reverse = tokenizer.encode(reverse)
            reverse_tensor = torch.tensor(reverse[-self.max_length:])

            response = (''.join(sample['response'].split(' ')) + PAD_WORD)[:256]
            self.response.append(response)
            response = tokenizer.encode(response)
            response_tensor = torch.tensor(response)

            goal = sample['goal']
            goal_full = []
            for g in goal:
                g_type, _, g_entity = g
                goal_full.append(g_type + ' ' + SEP_WORD + ' ' + g_entity)
            goal_full = (' ' + CLS_WORD + ' ').join(goal_full)
            goal_full = ''.join(goal_full.split(' '))
            goal_full = tokenizer.encode(goal_full)
            goal_tensor = torch.tensor(goal_full)

            entities_t = []
            knowledge = sample['knowledge']
            for k in knowledge:
                a, b, c = k
                a = ''.join(a.split(' '))
                b = ''.join(b.split(' '))
                c = ''.join(c.split(' '))
                entities_t.append((a, b, c))
            entities_t.sort(key=takeSecond)
            entities = []
            now = entities_t[0][0]
            temp = []
            for enti in entities_t:
                if enti[0] != now:
                    now = enti[0]
                    entities.append((now, temp))
                    temp = []
                temp.append((enti[1], enti[2]))
            strings = []
            entity_ids = []
            triple_ids = []

            for i, entity in enumerate(entities):
                if i + 1 >= 1024:
                    break

                # entity = self.knowledge[entity_id]

                string, triple_id = linearize_v2(tokenizer, entity)

                strings += string
                entity_ids += [i + 1] * len(string)
                triple_ids += triple_id

            position_ids = list(range(len(strings)))
            assert len(strings) == len(entity_ids) == len(triple_ids) == len(position_ids)

            if len(strings) >= 1024:
                input_ids = torch.LongTensor(strings[:self.max_enc_len])
                entity_ids = torch.LongTensor(entity_ids[:self.max_enc_len])
                triple_ids = torch.LongTensor(triple_ids[:self.max_enc_len])
                position_ids = torch.LongTensor(position_ids[:self.max_enc_len])
            else:
                input_ids = torch.LongTensor(strings + [self.pad_idx] * (self.max_enc_len - len(strings)))
                entity_ids = torch.LongTensor(entity_ids + [0] * (self.max_enc_len - len(strings)))
                triple_ids = torch.LongTensor(triple_ids + [0] * (self.max_enc_len - len(strings)))
                position_ids = torch.LongTensor(position_ids + [0] * (self.max_enc_len - len(strings)))

            emotion_tensor = torch.tensor([self.emotion_vocab.get(w) for w in sample['emotion'][:-1]], requires_grad=False).long()
            next_emotion = torch.tensor([self.emotion_vocab.get(sample['emotion'][-1])], requires_grad=False).long()

            self.sample_tensor.append([id_tensor, context_tensor, response_tensor, reverse_tensor, goal_tensor,
                                       input_ids, entity_ids, triple_ids, position_ids, emotion_tensor, next_emotion])
            self.len = idx + 1
            if idx >= self.n:
                break

        print('data size: ', self.len)

    def __getitem__(self, index):
        return self.sample_tensor[index]

    def __len__(self):
        return self.len


def collate_fn(data):
    id, context, response, reverse, goal, input_ids, entity_ids, triple_ids, position_ids, emotion, next_emo = zip(*data)
    # id, context, response, reverse, goal, emotion, next_emo = zip(*data)

    return {'id': torch.cat(id),
            'context': pad_sequence(context, batch_first=True),
            'response': pad_sequence(response, batch_first=True),
            'reverse': pad_sequence(reverse, batch_first=True),
            'goal': pad_sequence(goal, batch_first=True),
            'knowledge': pad_sequence(input_ids, batch_first=True),
            'entity_ids': pad_sequence(entity_ids, batch_first=True),
            'triple_ids': pad_sequence(triple_ids, batch_first=True),
            'position_ids': pad_sequence(position_ids, batch_first=True),
            'emotion': pad_sequence(emotion, batch_first=True),
            'next_emotion': pad_sequence(next_emo, batch_first=True)}

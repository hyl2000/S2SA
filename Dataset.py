from torch.utils.data import Dataset
from Utils import *
from torch.nn.utils.rnn import pad_sequence
import json
import re


def pad_knowledge(knowledge):
    for i in range(len(knowledge)):
        if len(knowledge[i]) > 50:
            knowledge[i] = knowledge[i][0:50]
        else:
            knowledge[i] = knowledge[i] + (50 - len(knowledge[i])) * [0]
    return knowledge


class Dataset(Dataset):
    def __init__(self, path, vocab2id, max_length=200, n=1E10):
        super(Dataset, self).__init__()

        self.emotion_vocab = {'pad': 0, '认同': 1, '不认同': 2, '开心': 3, '伤心': 4, '惊讶': 5, '好奇': 6, '中立': 7}

        self.max_length = max_length
        self.path = path

        self.response = []
        self.context = []

        self.vocab2id = vocab2id
        self.n = n

        self.sample_tensor = []
        self.samples = []
        self.len = 0
        self.load()

    def load(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                session = json.loads(line.strip(), encoding="utf-8")
                self.samples.append(session)
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

            contexts = ' [SEP] '.join(sample["history"][2:])
            contexts = contexts.split(' ')
            for i in range(len(contexts)):
                contexts[i] = re.sub('\d+', '<num>', contexts[i])

            while len(contexts) < 2:
                contexts = ['<nan>'] + contexts
            contexts = contexts[-self.max_length:]
            self.context.append(' '.join(contexts))
            context_tensor = torch.tensor(
                [self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in contexts],
                requires_grad=False).long()

            reverse = ' [SEP] '.join(sample["reverse"])
            reverse = reverse.split(' ')
            for i in range(len(reverse)):
                reverse[i] = re.sub('\d+', '<num>', reverse[i])

            while len(reverse) < 2:
                reverse = ['<nan>'] + reverse
            reverse = reverse[-self.max_length:]
            # self.reverse.append(' '.join(reverse))
            reverse_tensor = torch.tensor(
                [self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in reverse],
                requires_grad=False).long()

            response = (sample['response'].split(' ') + [EOS_WORD])[:80]
            for i in range(len(response)):
                response[i] = re.sub('\d+', '<num>', response[i])
            self.response.append(' '.join(response))
            response_tensor = torch.tensor(
                [self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in response],
                requires_grad=False).long()

            goal = sample['goal']
            goal_full = []
            for g in goal:
                g_type, _, g_entity = g
                goal_full.append(g_type + ' [SEP] ' + g_entity)
            goal_full = ' [CLS] '.join(goal_full)
            goal_full = goal_full.split(' ')
            goal_tensor = torch.tensor(
                [self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in goal_full],
                requires_grad=False).long()

            knowledge = sample['knowledge']
            knowledge_full = []
            for k in knowledge:
                a, b, c = k
                knowledge_full.append(a + ' [SEP] ' + b + ' [SEP] ' + c)
            knowledge_tensor = []
            for k in knowledge_full:
                temp = k.split(' ')
                knowledge_tensor.append([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in temp])
            knowledge_tensor = pad_knowledge(knowledge_tensor)
            knowledge_tensor = torch.tensor(knowledge_tensor, requires_grad=False).long()

            emotion_tensor = torch.tensor([self.emotion_vocab.get(w) for w in sample['emotion'][:-1]], requires_grad=False).long()
            next_emotion = torch.tensor([self.emotion_vocab.get(sample['emotion'][-1])], requires_grad=False).long()

            self.sample_tensor.append([id_tensor, context_tensor, response_tensor, reverse_tensor, goal_tensor, knowledge_tensor, emotion_tensor, next_emotion])
            self.len = idx + 1
            if idx >= self.n:
                break
        print('data size: ', self.len)

    def __getitem__(self, index):
        return self.sample_tensor[index]

    def __len__(self):
        return self.len


def collate_fn(data):
    id, context, response, reverse, goal, knowledge, emotion, next_emo = zip(*data)

    return {'id': torch.cat(id),
            'context': pad_sequence(context, batch_first=True),
            'response': pad_sequence(response, batch_first=True),
            'reverse': pad_sequence(reverse, batch_first=True),
            'goal': pad_sequence(goal, batch_first=True),
            'knowledge': pad_sequence(knowledge, batch_first=True),
            'emotion': pad_sequence(emotion, batch_first=True),
            'next_emotion': pad_sequence(next_emo, batch_first=True)}

from torch.utils.data import Dataset
from Utils import *
from torch.nn.utils.rnn import pad_sequence
import json
import re


class Dataset(Dataset):
    def __init__(self, path, vocab2id, max_length=200, n=1E10):
        super(Dataset, self).__init__()

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
            if sample["response"].startswith('['):
                sample["response"] = sample["response"][4:]
            contexts = " [SEP] ".join(sample["history"][2:])
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

            response = (sample['response'].split(' ') + [EOS_WORD])[:80]
            for i in range(len(response)):
                response[i] = re.sub('\d+', '<num>', response[i])
            self.response.append(' '.join(response))
            response_tensor = torch.tensor(
                [self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in response],
                requires_grad=False).long()

            self.sample_tensor.append([id_tensor, context_tensor, response_tensor])
            self.len = idx + 1
            if idx >= self.n:
                break
        print('data size: ', self.len)

    def __getitem__(self, index):
        return self.sample_tensor[index]

    def __len__(self):
        return self.len


def collate_fn(data):
    id, context, response = zip(*data)

    return {'id': torch.cat(id),
            'context': pad_sequence(context, batch_first=True),
            'response': pad_sequence(response, batch_first=True)}

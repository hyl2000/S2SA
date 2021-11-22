from Dataset import Dataset, collate_fn
from torch import optim
from DefaultTrainer import DefaultTrainer
import torch.backends.cudnn as cudnn
import argparse
import torch
import os
from Model import S2SA
from dataset.Utils import load_vocab, nltk_tokenizer, nltk_detokenizer, load_default, split_data, init_seed, init_params

base_output_path = 'holl_output/S2SA_output/'
dataset = 'holl_e'
data_path = 'holl_e/'
vocab_dict = 'holl_e/holl_e.vocab'
dir_path = os.path.dirname(os.path.realpath(__file__))
embedding_size = 300
hidden_size = 256
background_len = 300
min_vocab_freq = 0
cache_size = 10000
vocab_size = 25000


def train(args):
    # if torch.cuda.is_available():
    #     torch.distributed.init_process_group(backend='NCCL', init_method='env://')

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)

    batch_size = 64

    output_path = base_output_path

    vocab2id, id2vocab, id2freq = load_vocab(vocab_dict, t=min_vocab_freq, vocab_size=vocab_size)

    tokenizer = nltk_tokenizer()
    detokenizer = nltk_detokenizer()

    if os.path.exists(data_path + 'train_samples.pkl'):
        train_samples = torch.load(data_path + 'train_samples.pkl')
        query = torch.load(data_path + 'query.pkl')
        passage = torch.load(data_path + 'passage.pkl')
    else:
        samples, query, _, passage = load_default(data_path + dataset + '.answer', data_path + dataset + '.passage',
                                                  data_path + dataset + '.pool', data_path + dataset + '.qrel',
                                                  data_path + dataset + '.query',
                                                  data_path + dataset + '.reformulation.query', tokenizer)
        train_samples, dev_samples, test_samples = split_data(data_path + dataset + '.split', samples)
        torch.save(train_samples, data_path + 'train_samples.pkl')
        torch.save(dev_samples, data_path + 'dev_samples.pkl')
        torch.save(test_samples, data_path + 'test_samples.pkl')
        torch.save(query, data_path + 'query.pkl')
        torch.save(passage, data_path + 'passage.pkl')

    model = S2SA(embedding_size, hidden_size, vocab2id, id2vocab, max_dec_len=70, beam_width=1)
    init_params(model)
    # file = output_path + 'model/' + '3' + '.pkl'
    # model.load_state_dict(torch.load(file))

    model_optimizer = optim.Adam(model.parameters())

    trainer = DefaultTrainer(model, tokenizer, detokenizer, args.local_rank)

    for i in range(20):
        count = 0
        while count < len(train_samples):
            train_dataset = Dataset(train_samples[count:count + cache_size], query, passage, vocab2id, background_len)
            count += cache_size
            trainer.train_epoch('mle_train', train_dataset, collate_fn, batch_size, i, model_optimizer)
            del train_dataset
        trainer.serialize(i, output_path=output_path)


def test(args):
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)

    batch_size = 32

    output_path = base_output_path

    vocab2id, id2vocab, id2freq = load_vocab(vocab_dict, t=min_vocab_freq, vocab_size=vocab_size)

    tokenizer = nltk_tokenizer()
    detokenizer = nltk_detokenizer()

    if os.path.exists(data_path + 'test_samples.pkl'):
        test_samples = torch.load(data_path + 'test_samples.pkl')
        query = torch.load(data_path + 'query.pkl')
        passage = torch.load(data_path + 'passage.pkl')
    else:
        samples, query, _, passage = load_default(data_path + dataset + '.answer', data_path + dataset + '.passage',
                                                  data_path + dataset + '.pool', data_path + dataset + '.qrel',
                                                  data_path + dataset + '.query',
                                                  data_path + dataset + '.reformulation.query', tokenizer)
        train_samples, dev_samples, test_samples = split_data(data_path + dataset + '.split', samples)
        torch.save(train_samples, data_path + 'train_samples.pkl')
        torch.save(dev_samples, data_path + 'dev_samples.pkl')
        torch.save(test_samples, data_path + 'test_samples.pkl')
        torch.save(query, data_path + 'query.pkl')
        torch.save(passage, data_path + 'passage.pkl')

    test_dataset = Dataset(test_samples, query, passage, vocab2id, background_len)

    for i in range(20):
        print('epoch', i)
        file = output_path + 'model/' + str(i) + '.pkl'
        if os.path.exists(file):
            model = S2SA(embedding_size, hidden_size, vocab2id, id2vocab, max_dec_len=70, beam_width=1)
            model.load_state_dict(torch.load(file))
            trainer = DefaultTrainer(model, tokenizer, detokenizer, None)
            trainer.test('test', test_dataset, collate_fn, batch_size, i, output_path=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    if args.mode == 'test':
        test(args)
    elif args.mode == 'train':
        train(args)

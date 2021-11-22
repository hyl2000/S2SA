from Dataset import *
from torch import optim
from DefaultTrainer import *
import torch.backends.cudnn as cudnn
import argparse
import os
from Model import *
from Utils import *
import torch

cudaid = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaid)

base_output_path = 'output/'

embedding_size = 300
hidden_size = 256
knowledge_len = 300
min_vocab_freq = 50


def train(args):
    torch.cuda.set_device(args.local_rank)
    # if torch.cuda.is_available():
    #     torch.distributed.init_process_group(backend='nccl')

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)

    batch_size = 32

    output_path = base_output_path
    dataset = 'sample'
    data_path = 'data/'
    print('go...')
    vocab2id, id2vocab = load_vocab('data/vocab.txt', t=min_vocab_freq)
    print('load_vocab done')

    train_dataset = Dataset(data_path + dataset + '.'+"train.txt", vocab2id, knowledge_len)
    print('build data done')
    model = S2SA(embedding_size, hidden_size, vocab2id, id2vocab, max_dec_len=70, beam_width=1)
    print('build model done')
    init_params(model, escape='embedding')
    print('init_params done')
    model_optimizer = optim.Adam(model.parameters())

    trainer = DefaultTrainer(model, args.local_rank)
    print('start training...')
    for i in range(20):
        print('#', i)
        if i == 5:
            train_embedding(model)
        trainer.train_epoch('mle_train', train_dataset, collate_fn, batch_size, i, model_optimizer)
        trainer.serialize(i, output_path=output_path)


def test(args, beam_width):
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)

    batch_size = 64

    output_path = base_output_path

    dataset = 'sample'
    data_path = 'data/'
    vocab2id, id2vocab = load_vocab('data/vocab.txt', t=min_vocab_freq)

    test_dataset = Dataset(data_path + dataset + '.'+"test.txt", vocab2id, knowledge_len)

    model = S2SA(embedding_size, hidden_size, vocab2id, id2vocab, max_dec_len=70, beam_width=beam_width)
    trainer = DefaultTrainer(model, None)
    trainer.test('test', test_dataset, collate_fn, batch_size, 0, output_path=output_path)

    for i in range(20):
        print('epoch', i)
        file = output_path + 'model/' + str(i) + '.pkl'

        if os.path.exists(file):
            model = S2SA(embedding_size, hidden_size, vocab2id, id2vocab, max_dec_len=70, beam_width=beam_width)
            model.load_state_dict(torch.load(file))
            trainer = DefaultTrainer(model, None)
            trainer.test('test', test_dataset, collate_fn, batch_size, 100 + i, output_path=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--mode", default='test', type=str)
    parser.add_argument("--beam_width", default=5, type=int)
    args = parser.parse_args()

    # test(args)

    if args.mode == 'test':
        test(args, args.beam_width)
    elif args.mode == 'train':
        train(args)

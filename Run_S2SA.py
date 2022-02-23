from Dataset import *
from torch import optim
from DefaultTrainer import *
import torch.backends.cudnn as cudnn
import argparse
import Constants
import os
from Model import *
from Utils import *
import torch
from tqdm import tqdm, trange
from transformers import AdamW, Adafactor, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
cudaid = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaid)

base_output_path = 'output/'

embedding_size = 300
hidden_size = 512
knowledge_len = 512
min_vocab_freq = 50


def train(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    generator = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)

    batch_size = 1

    output_path = base_output_path
    dataset = 'sample'
    data_path = 'data/'
    train_path = data_path + dataset + '.' + "xtrain.txt"
    valid_path = data_path + dataset + '.' + "xdev.txt"
    print('go...')
    vocab2id, id2vocab, entity2id, relation2id = load_vocab('data/vocab.txt', 'data/entities.txt', 'data/relations.txt', t=min_vocab_freq)
    print('load_vocab done')

    train_dataset = Dataset(train_path, vocab2id, entity2id, relation2id, batch_size, model_name, valid_path, knowledge_len)
    print('build data done')
    model = S2SA(embedding_size, hidden_size, vocab2id, id2vocab, entity2id, relation2id, generator, max_dec_len=70, beam_width=1)
    model_optimizer = optim.Adam(model.parameters())
    if args.load_epoch != 0:
        file = output_path + 'model/' + str(args.load_epoch) + '.pkl'
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint["state_dict"])
        model_optimizer.load_state_dict(checkpoint["optimizer_state"])
        args.load_epoch += 1
    print('build model done')
    init_params(model, escape='embedding')
    print('init_params done')

    trainer = DefaultTrainer(model, args.local_rank)
    print('start training main model...')
    for i in range(args.load_epoch, args.max_epoch):
        print('#', i)
        if i == 5:
            train_embedding(model)
        trainer.train_epoch('mle_train', train_dataset, collate_fn, batch_size, i, model_optimizer)
        trainer.serialize(model_optimizer, i, output_path=output_path)


def test(args, beam_width):
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)

    batch_size = 2

    output_path = base_output_path

    dataset = 'sample'
    data_path = 'data/'
    # model_name = 't5-base'
    generator = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    vocab2id, id2vocab, entity2id, relation2id = load_vocab('data/vocab.txt', 'data/entities.txt', 'data/relations.txt',
                                                            t=min_vocab_freq)

    test_dataset = Dataset(data_path + dataset + '.'+"xtest.txt", vocab2id, entity2id, relation2id, batch_size, model_name, max_length=knowledge_len)

    for i in range(args.max_epoch):
        print('epoch', i)
        file = output_path + 'model/' + str(i) + '.pkl'

        if os.path.exists(file):
            model = S2SA(embedding_size, hidden_size, vocab2id, id2vocab, entity2id, relation2id, generator, max_dec_len=70, beam_width=beam_width)
            checkpoint = torch.load(file)
            model.load_state_dict(checkpoint["state_dict"])
            trainer = DefaultTrainer(model, None)
            trainer.test('test', test_dataset, collate_fn, batch_size, 100 + i, output_path=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--mode", default='train', type=str)
    parser.add_argument("--beam_width", default=5, type=int)
    parser.add_argument("--load_epoch", default=0, type=int)
    parser.add_argument("--max_epoch", default=20, type=int)
    args = parser.parse_args()

    # test(args)

    if args.mode == 'test':
        test(args, args.beam_width)
    elif args.mode == 'train':
        train(args)

'''
    Hugging face Examples::

    >> > from transformers import T5Tokenizer, T5ForConditionalGeneration

    >> > tokenizer = T5Tokenizer.from_pretrained('t5-small')
    >> > model = T5ForConditionalGeneration.from_pretrained('t5-small')

    >> >  # training
    >> > input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
    >> > labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
    >> > outputs = model(input_ids=input_ids, labels=labels)
    >> > loss = outputs.loss
    >> > logits = outputs.logits

    >> >  # inference
    >> > input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you",
                               return_tensors="pt").input_ids  # Batch size 1
    >> > outputs = model.generate(input_ids)
    >> > print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    >> >  # studies have shown that owning a dog is good for you.
'''
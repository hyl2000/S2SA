from Dataset import *
from torch import optim
from DefaultTrainer import *
import torch.backends.cudnn as cudnn
import argparse
import os
from Model import *
from Utils import *
import torch
from tqdm import tqdm, trange

cudaid = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaid)

base_output_path = 'output/'

embedding_size = 300
hidden_size = 256
knowledge_len = 300
min_vocab_freq = 50


def train_GCN(model, train_data, test_graph, valid_triplets, all_triplets):
    use_cuda = torch.cuda.is_available()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    reg_ratio = 1e-2
    best_mrr = 0
    grad_norm = 1.0
    evaluate_every = 500

    print(model)

    if use_cuda:
        model.cuda()

    for epoch in trange(1, (20 + 1), desc='Epochs', position=0):

        model.train()
        optimizer.zero_grad()

        if use_cuda:
            device = torch.device('cuda')
            train_data.to(device)

        entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
        loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) + reg_ratio * model.reg_loss(
            entity_embedding)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()

        if epoch % evaluate_every == 0:

            tqdm.write("Train Loss {} at epoch {}".format(loss, epoch))

            if use_cuda:
                model.cpu()

            model.eval()
            entity_embedding = model(test_graph.entity, test_graph.edge_index, test_graph.edge_type,
                                     test_graph.edge_norm)
            valid_mrr = calc_mrr(entity_embedding, model.relation_embedding, valid_triplets, all_triplets, hits=[1, 3, 10])

            if valid_mrr > best_mrr:
                best_mrr = valid_mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           './output/model/best_mrr_model.pth')

            if use_cuda:
                model.cuda()


def train(args):
    if torch.cuda.is_available():
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
    train_path = data_path + dataset + '.' + "xtrain.txt"
    valid_path = data_path + dataset + '.' + "xdev.txt"
    print('go...')
    vocab2id, id2vocab, entity2id, relation2id = load_vocab('data/vocab.txt', 'data/entities.txt', 'data/relations.txt', t=min_vocab_freq)
    print('load_vocab done')

    train_dataset = Dataset(train_path, vocab2id, entity2id, relation2id, batch_size, valid_path, knowledge_len)
    print('build data done')
    GCN = RGCN(len(entity2id), len(relation2id), num_bases=4, dropout=0.5)
    test_graph = build_test_graph(len(entity2id), len(relation2id), train_dataset.build_graph_data)
    print('start training GCN...')
    train_GCN(GCN, train_dataset.GCN_train_sample, test_graph, train_dataset.GCN_valid_sample, train_dataset.all_triplets)
    checkpoint = torch.load('./output/model/best_mrr_model.pth')
    GCN.load_state_dict(checkpoint['state_dict'])
    print('GCN training finished')
    model = S2SA(embedding_size, hidden_size, vocab2id, id2vocab, entity2id, relation2id, max_dec_len=70, beam_width=1)
    print('build model done')
    init_params(model, escape='embedding')
    print('init_params done')
    model_optimizer = optim.Adam(model.parameters())

    trainer = DefaultTrainer(model, args.local_rank)
    print('start training main model...')
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
    vocab2id, id2vocab, entity2id, relation2id = load_vocab('data/vocab.txt', 'data/entities.txt', 'data/relations.txt',
                                                            t=min_vocab_freq)

    test_dataset = Dataset(data_path + dataset + '.'+"xtest.txt", vocab2id, entity2id, relation2id, batch_size, knowledge_len)

    # model = S2SA(embedding_size, hidden_size, vocab2id, id2vocab, entity2id, relation2id, max_dec_len=70, beam_width=beam_width)
    # trainer = DefaultTrainer(model, None)
    # trainer.test('test', test_dataset, collate_fn, batch_size, 0, output_path=output_path)

    for i in range(20):
        print('epoch', i)
        file = output_path + 'model/' + str(i) + '.pkl'

        if os.path.exists(file):
            model = S2SA(embedding_size, hidden_size, vocab2id, id2vocab, entity2id, relation2id, max_dec_len=70, beam_width=beam_width)
            model.load_state_dict(torch.load(file))
            trainer = DefaultTrainer(model, None)
            trainer.test('test', test_dataset, collate_fn, batch_size, 100 + i, output_path=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--mode", default='train', type=str)
    parser.add_argument("--beam_width", default=5, type=int)
    args = parser.parse_args()

    # test(args)

    if args.mode == 'test':
        test(args, args.beam_width)
    elif args.mode == 'train':
        train(args)

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from Utils import *
import os
import sys


def train_embedding(model):
    for name, param in model.named_parameters():
        if 'embedding' in name:
            print('requires_grad', name, param.size())
            param.requires_grad = True


def init_params(model, escape=None):
    for name, param in model.named_parameters():
        if escape is not None and escape in name:
            print('no_init', name, param.size())
            continue
        print('init', name, param.size())
        if param.data.dim() > 1:
            xavier_uniform_(param.data)


class DefaultTrainer(object):
    def __init__(self, model, local_rank):
        super(DefaultTrainer, self).__init__()
        self.local_rank = local_rank

        if local_rank is not None and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model
        self.eval_model = self.model

        # if torch.cuda.is_available() and local_rank is not None:
        #     print("GPU ", self.local_rank)
        #     self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    def train_batch(self, epoch, data, method, optimizer):
        optimizer.zero_grad()
        loss, count = self.model(data, method=method)

        if isinstance(loss, tuple) or isinstance(loss, list):
            closs = [l.mean().cpu().item() for l in loss]
            # loss = torch.cat([l.mean().view(1) for l in loss]).sum()
            loss = torch.cat(loss, dim=-1).mean()
        else:
            loss = loss.mean()
            closs = [loss.cpu().item()]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
        optimizer.step()
        return closs, count

    def serialize(self, epoch, output_path):
        if self.local_rank != 0:
            return
        output_path = os.path.join(output_path, 'model/')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        torch.save(self.eval_model.state_dict(), os.path.join(output_path, '.'.join([str(epoch), 'pkl'])))

    def train_epoch(self, method, train_dataset, train_collate_fn, batch_size, epoch, optimizer):
        self.model.train()
        train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=batch_size,
                                                   shuffle=True)
        start_time = time.time()
        count_batch = 0
        count_num = 0
        for j, data in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                data = data_cuda
            count_batch += 1

            bloss, count = self.train_batch(epoch, data, method=method, optimizer=optimizer)
            count_num += count
            emo_acc = count_num/(count_batch * batch_size)

            if j >= 0 and j % 100 == 0:
                elapsed_time = time.time() - start_time
                print('Method', method, 'Epoch', epoch, 'Batch ', count_batch, 'Loss ', bloss, 'Emo_acc ', emo_acc, 'Time ', elapsed_time)
                sys.stdout.flush()
            del bloss

        # elapsed_time = time.time() - start_time
        # print(method + ' ', epoch, 'time ', elapsed_time)
        sys.stdout.flush()

    def predict(self, method, dataset, collate_fn, batch_size, epoch, output_path):
        self.eval_model.eval()

        with torch.no_grad():
            test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                                     num_workers=0)

            systems = []
            ref_path = None
            count_total = 0
            total = 0
            for k, data in enumerate(test_loader, 0):
                if torch.cuda.is_available():
                    data_cuda = dict()
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data_cuda[key] = value.cuda()
                        else:
                            data_cuda[key] = value
                    data = data_cuda

                indices, count = self.eval_model(data, method=method)
                count_total += count
                sents = self.eval_model.to_sentence(data, indices)

                remove_duplicate(sents)

                for i in range(len(data['id'])):
                    total += 1
                    idx = data['id'][i].item()
                    systems.append(' '.join(sents[i]) + '\t' + dataset.response[idx][:-6])

            output_path = os.path.join(output_path, 'result_raw/')
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            output_path = os.path.join(output_path, str(epoch) + '.txt')
            file = codecs.open(output_path, "w", "utf-8")
            for i in range(len(systems)):
                file.write(systems[i] + os.linesep)
            file.close()

            emo_acc = count_total / total
            print('Emo_acc: ', emo_acc)
        return output_path

    def test(self, method, dataset, collate_fn, batch_size, epoch, output_path):
        with torch.no_grad():
            run_file = self.predict(method, dataset, collate_fn, batch_size, epoch, output_path)
        print("finish")
        sys.stdout.flush()
        return 0

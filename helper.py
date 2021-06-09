import os
import json
import torch
import torch.autograd as autograd
import numpy as np
import logging
import heapq
import random


class TopKHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = []

    def push(self, elem):
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem > topk_small:
                heapq.heapreplace(self.data, elem)

    def topk(self):
        return [x for x in reversed([heapq.heappop(self.data) for x in range(len(self.data))])]


def override_config(args):
    """
    Override model and data configuration.
    """

    with open(os.path.join(args.init_checkpoint, 'configs.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']


def save_model(model, ssvae_model, optimizer, save_variable_list, args, is_best_model=False):
    """
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate.
    """

    save_path = "%s/best/" % args.save_path if is_best_model else args.save_path
    argparse_dict = vars(args)
    with open(os.path.join(save_path, 'configs.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    # Save KG embedding model
    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(save_path, 'checkpoint')
    )

    # Save ssvae model
    torch.save({
        'model_state_dict': ssvae_model.state_dict()},
        os.path.join(save_path, 'ssvae_checkpoint')
    )
    if model.entity_embedding is not None:
        entity_embedding = model.entity_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(save_path, 'entity_embedding'),
            entity_embedding
        )

        relation_embedding = model.relation_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(save_path, 'relation_embedding'),
            relation_embedding
        )


def read_triple(file_path, entity2id=None, relation2id=None):
    """
    Read triples and map them into ids.
    """

    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            if entity2id is None or relation2id is None:
                triples.append((int(h), int(r), int(t)))
            else:
                triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def set_logger(args):
    """
    Write logs to checkpoint and console.
    """

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO if not args.debug else logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if not args.debug else logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def set_logger_(args, detect=True):
    """
    Write logs to checkpoint and console.
    """

    if detect:
        log_file = os.path.join(args.init_path, 'detect_error.log')
    else:
        log_file = os.path.join(args.init_path, 'repair_error.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def is_better_metric(best_metrics, cur_metrics):
    if best_metrics is None:
        return True
    if best_metrics[-1]['MRR'] < cur_metrics[-1]['MRR']:
        return True
    return False


def log_metrics(mode, step, metrics):
    """
    Print the evaluation logs.
    """

    for metric in metrics:
        if 'name' in metric:
            logging.info("results from %s" % metric['name'])
        for m in [x for x in metric if x != "name"]:
            logging.info('%s %s at step %d: %f' % (mode, m, step, metric[m]))


def worker_init(worker_init):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def binary_cross_entropy(r, x):
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)


def RotatE_Trans(ent, rel, is_hr):
    re_ent, im_ent = ent
    re_rel, im_rel = rel
    if is_hr:  # ent == head
        re = re_ent * re_rel - im_ent * im_rel
        im = re_ent * im_rel + im_ent * re_rel
    else:  # ent == tail
        re = re_rel * re_ent + im_rel * im_ent
        im = re_rel * im_ent - im_rel * re_ent
    return re, im

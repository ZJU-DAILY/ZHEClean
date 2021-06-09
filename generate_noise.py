import torch
from tqdm import tqdm
import random
import os
import numpy as np
import argparse
from collections import defaultdict
from helper import *


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Generate error.',
        usage='generate_error.py [<args>] [-h | --help]'
    )

    parser.add_argument('--seed', default=2021, type=int)

    args = parser.parse_args(args)

    return args


def generate_noise(data_path, rate_list):
    with open(os.path.join(data_path, 'entities.dict')) as fin:
        entity2id = {}
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(data_path, 'relations.dict')) as fin:
        relation2id = {}
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    nentity = len(entity2id)
    nrelation = len(relation2id)
    train_triples = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
    all_triples = train_triples + valid_triples + test_triples
    true_head, true_tail, true_relation = defaultdict(lambda: set()), defaultdict(lambda: set()), defaultdict(
        lambda: set())
    relation2tail, relation2head = defaultdict(lambda: set()), defaultdict(lambda: set())
    for h, r, t in all_triples:
        true_head[(r, t)].add(h)
        true_tail[(h, r)].add(t)
        true_relation[(h, t)].add(r)
        relation2tail[r].add(t)
        relation2head[r].add(h)

    # Reducing false negative labels from TransH.
    tph = {}
    hpt = {}
    for i in range(nrelation):
        tph[i] = len(relation2tail[i]) / len(relation2head[i])
        hpt[i] = len(relation2head[i]) / len(relation2tail[i])

    # Generate error for train and test
    print('Generate error...')
    for triple in [train_triples, test_triples]:
        for rate in rate_list:
            noise_triples = set()
            num = len(triple) * rate // 100
            with tqdm(total=num) as pbar:
                pbar.set_description('Processing:')
                while len(noise_triples) < num:
                    h, r, t = random.choice(triple)
                    mode = np.random.randint(low=0, high=3)
                    if mode < 2:
                        temp = np.random.binomial(1, tph[r] / (tph[r] + hpt[r]))
                        if temp == 1:
                            h_ = random.choice(list(set(entity2id.values()) - true_head[(r, t)]))
                            noise_triples.add((h_, r, t))
                        else:
                            t_ = random.choice(list(set(entity2id.values()) - true_tail[(h, r)]))
                            noise_triples.add((h, r, t_))
                    else:
                        r_ = random.choice(list(set(relation2id.values()) - true_relation[(h, t)]))
                        noise_triples.add((h, r_, t))
                    pbar.update()

            noise_triples = list(noise_triples)
            if triple is train_triples:
                save_file = os.path.join(data_path, 'noise_' + str(rate) + '.txt')
            else:
                save_file = os.path.join(data_path, 'test_negative_' + str(rate) + '.txt')
            np.savetxt(save_file, noise_triples, fmt='%d', delimiter='\t', newline='\n')


if __name__ == '__main__':
    args = parse_args()

    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    rate_list = [10, 20, 30, 40, 50, 60, 80, 100]

    data_path_list = ['WN18RR', 'FB15K-237', 'YAGO3-10-DR']

    for data_path in data_path_list:
        print('Data: {}'.format(data_path))
        cur_data_path = os.path.join('./dataset', data_path)
        generate_noise(data_path=cur_data_path, rate_list=rate_list)

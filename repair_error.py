import argparse
import os
import torch
import json
import random
from tqdm import tqdm
import numpy as np
import logging
from model import KGEModel
from helper import read_triple, set_logger_


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Repair error.',
        usage='repair_error.py [<args>] [-h | --help]'
    )

    parser.add_argument('--data_name', type=str, default='WN18RR')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--init_path', default=None)
    parser.add_argument('--model', default='RotatE', type=str)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)

    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('--train_error_rate', default=20, type=int)

    parser.add_argument('--alpha', type=float, default=0.9)

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--seed', default=2021, type=int)

    args = parser.parse_args(args)

    return args


def evaluate_repair(clean_triples, all_true_triples, num_errors, error_index_set):
    TP = 0
    num_update = 0

    for index, cur_triple in enumerate(clean_triples):
        if index in error_index_set:
            num_update += 1
            if cur_triple in all_true_triples:
                TP += 1

    precision = TP / num_update
    recall = TP / num_errors
    f1 = (2 * precision * recall) / (precision + recall)
    logging.info('Precision: {}'.format(precision))
    logging.info('Recall: {}'.format(recall))
    logging.info('F1 score: {}'.format(f1))


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

    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[args.data_name]
    args.data_path = config['data_path']
    args.hidden_dim = config['hidden_dim']
    if args.data_name == 'NELL27K':
        args.init_path = './checkpoint/{}-{}-soft'.format(args.data_name, args.model)
    else:
        args.init_path = './checkpoint/{}-{}-{}-soft'.format(args.data_name, args.model, args.train_error_rate)
    if args.model == 'RotatE':
        args.double_entity_embedding = True

    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = {}
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = {}
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    set_logger_(args, detect=False)
    logging.info(args)

    args.nentity = nentity
    args.nrelation = nrelation

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=0,
        args=args,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
    )

    checkpoint = torch.load(os.path.join(args.init_path, 'checkpoint'))
    kge_model.load_state_dict(checkpoint['model_state_dict'])
    kge_model = kge_model.cuda()
    entity_embedding = np.load(os.path.join(args.init_path, 'entity_embedding.npy'))
    relation_embedding = np.load(os.path.join(args.init_path, 'relation_embedding.npy'))
    entity_embedding = torch.from_numpy(entity_embedding).to(kge_model.entity_embedding.device)
    relation_embedding = torch.from_numpy(relation_embedding).to(kge_model.relation_embedding.device)
    kge_model.entity_embedding.data[:entity_embedding.size(0)] = entity_embedding
    kge_model.relation_embedding.data[:relation_embedding.size(0)] = relation_embedding

    kge_model.eval()
    model_func = kge_model.get_model_func()

    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    # Evaluate repair using all true triples.
    all_true_triples = set(train_triples + valid_triples + test_triples)

    if args.data_name == 'NELL27K':
        file_name = 'noise.txt'
    else:
        file_name = 'noise_{}.txt'.format(str(args.train_error_rate))
    noise_triples = np.loadtxt(os.path.join(args.data_path, file_name), dtype=np.int32)
    noise_triples = [tuple(x) for x in noise_triples.tolist()]
    all_train_triples = train_triples + noise_triples

    in_triples = {}
    out_triples = {}
    # Calculate in and out triples.
    for triple in all_train_triples:
        h, r, t = triple
        if h not in out_triples:
            out_triples[h] = []

        out_triples[h].append([r, t])

        if t not in in_triples:
            in_triples[t] = []

        in_triples[t].append([h, r])
    for test_error_rate in [10, 20, 30, 40, 50]:
        num_errors = 0
        if args.data_name == 'NELL27K':
            logging.info('Test')
            file_path = os.path.join(args.init_path, 'error_triples.txt')
            output_file_path = os.path.join(args.init_path, 'clean_triples.txt')
            error_file_path = os.path.join(args.data_path, 'test_negative.txt')
        else:
            logging.info('Test error rate: {}'.format(test_error_rate))
            file_path = os.path.join(args.init_path, 'error_triples_{}.txt'.format(str(test_error_rate)))
            output_file_path = os.path.join(args.init_path, 'clean_triples_{}.txt'.format(str(test_error_rate)))
            error_file_path = os.path.join(args.data_path, 'test_negative_{}.txt'.format(str(test_error_rate)))
        error_triples = np.loadtxt(file_path, dtype=np.int32)
        ground_truth_error_triples = read_triple(error_file_path)
        ground_truth_error_triples = set(ground_truth_error_triples)
        num_errors = len(ground_truth_error_triples)
        ground_truth_error_triples_index = set()

        use_outer_power = True

        logging.info('Using outer power: {}'.format(use_outer_power))
        clean_triples = []
        for triple_index, triple in enumerate(tqdm(error_triples)):
            h, r, t = triple
            temp = (h, r, t)
            if temp in ground_truth_error_triples:
                ground_truth_error_triples_index.add(triple_index)
            candidate_triples = [(_, r, t) for _ in range(nentity)] + [(h, r, _) for _ in range(nentity)] + [(h, _, t) for _ in range(nrelation)]

            inner_power = torch.zeros(2 * nentity + nrelation)
            i = 0
            while i < len(candidate_triples):
                j = min(i + 4096, len(candidate_triples))
                sample = torch.LongTensor(candidate_triples[i: j]).cuda()
                h_embedding = torch.index_select(entity_embedding, 0, sample[:, 0])
                r_embedding = torch.index_select(relation_embedding, 0, sample[:, 1])
                t_embedding = torch.index_select(entity_embedding, 0, sample[:, 2])
                s = model_func[kge_model.model_name](h_embedding, r_embedding, t_embedding, 'single', True)
                score = (-torch.norm(s, p=2, dim=1)).view(-1).detach().cpu()
                inner_power[i: j] = torch.sigmoid(score)
                i = j
            inner_power[t] = inner_power[h + nentity] = 0.0
            all_power = inner_power
            if use_outer_power:
                _, outer_power_index = torch.topk(inner_power, k=5)
                outer_power = torch.zeros(2 * nentity + nrelation)
                for c_index in outer_power_index:
                    candidate = candidate_triples[c_index]
                    c_h, c_r, c_t = candidate

                    h_embedding = entity_embedding[c_h]
                    r_embedding = relation_embedding[c_r]
                    t_embedding = entity_embedding[c_t]

                    in_score = 0.0
                    out_score = 0.0

                    if c_h in in_triples:
                        temp_triples = torch.LongTensor(in_triples[c_h]).cuda()

                        in_h = temp_triples[:, 0]
                        in_r = temp_triples[:, 1]

                        in_h_embedding = torch.index_select(entity_embedding, 0, in_h)
                        in_r_embedding = torch.index_select(relation_embedding, 0, in_r)
                        in_t_embedding = t_embedding

                        if args.model == 'TransE':
                            in_r_embedding = in_r_embedding + r_embedding
                        elif args.model == 'RotatE':
                            in_r_embedding = in_r_embedding * r_embedding

                        s = model_func[kge_model.model_name](in_h_embedding, in_r_embedding, in_t_embedding, 'single', True)
                        score = (-torch.norm(s, p=2, dim=-1)).view(-1).detach().cpu()

                        in_score = torch.sigmoid(torch.mean(score))

                    if c_t in out_triples:
                        temp_triples = torch.LongTensor(out_triples[c_t]).cuda()
                        out_r = temp_triples[:, 0]
                        out_t = temp_triples[:, 1]
                        out_t_embedding = torch.index_select(entity_embedding, 0, out_t)
                        out_h_embedding = h_embedding
                        out_r_embedding = torch.index_select(relation_embedding, 0, out_r)

                        if args.model == 'TransE':
                            out_r_embedding = out_r_embedding + r_embedding
                        elif args.model == 'RotatE':
                            out_r_embedding = out_r_embedding * r_embedding

                        s = model_func[kge_model.model_name](out_h_embedding, out_r_embedding, out_t_embedding, 'single', True)
                        score = (-torch.norm(s, p=2, dim=-1)).view(-1).detach().cpu()

                        out_score = torch.sigmoid(torch.mean(score))

                    if in_score == 0.0 and out_score == 0.0:
                        outer_power[c_index] = inner_power[c_index]
                    elif in_score == 0.0:
                        outer_power[c_index] = out_score
                    elif out_score == 0.0:
                        outer_power[c_index] = in_score
                    else:
                        outer_power[c_index] = (in_score + out_score) / 2

                    all_power[c_index] = args.alpha * inner_power[c_index] + (1 - args.alpha) * outer_power[c_index]

            index = torch.argmax(all_power).item()

            if index < nentity:
                clean_triples.append((index, r, t))
            elif index < 2 * nentity:
                clean_triples.append((h, r, index - nentity))
            else:
                clean_triples.append((h, index - 2 * nentity, t))

        np.savetxt(output_file_path, clean_triples, fmt='%d', delimiter='\t')
        evaluate_repair(clean_triples, all_true_triples, num_errors, ground_truth_error_triples_index)

        if args.data_name == 'NELL27K':
            break





import argparse
import json
import logging
import os
import sys
sys.path.append('./semi-supervised')
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
from helper import *
from inference import SVI, DeterministicWarmup, ImportanceWeightedSampler
from models import DeepGenerativeModel
from models import AuxiliaryDeepGenerativeModel


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models.',
        usage='run.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', default=True)
    parser.add_argument('--do_valid', default=True)
    parser.add_argument('--do_test', default=True)

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--noise_rate', type=int, default=20,
                        help='Noisy triples ratio of true train triples.')
    parser.add_argument('--mode', type=str, default='soft', choices=['none', 'soft'])
    parser.add_argument('--update_steps', type=int, default=50000, help='update confidence every xx steps.')
    parser.add_argument('--max_rate', type=int, default=0, help='DO NOT MANUALLY SET')

    # VAE
    parser.add_argument('--vae_model', type=str, default='ADM', choices=['DGM', 'ADM'])
    parser.add_argument('--z_dim', type=int, default=10)
    parser.add_argument('--a_dim', type=int, default=10)
    parser.add_argument('--h_dim', default='[20, 20]')
    parser.add_argument('--ssvae_steps', type=int, default=3000)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--loss_beta', type=float, default=2.0, help='0.1->2.0')
    parser.add_argument('--mc', type=int, default=5)
    parser.add_argument('--iw', type=int, default=5)

    parser.add_argument('--data_name', type=str, default='WN18RR')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('-save', '--save_path', default='./checkpoint/WN18RR', type=str)
    parser.add_argument('--model', default='RotatE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=512, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=6.0, type=float)  # used for modeling in TransE, RotatE, pRotatE
    parser.add_argument('-adv', '--negative_adversarial_sampling', default=True)
    parser.add_argument('-a', '--adversarial_temperature', default=0.5,
                        type=float)  # used for negative_adversarial_sampling only

    parser.add_argument('-b', '--batch_size', default=512, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)

    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')

    parser.add_argument('-lr', '--learning_rate', default=0.00005, type=float)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--init_embedding', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=1000, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--label_smoothing', default=0.1, type=float, help="used for bceloss ")

    parser.add_argument('--seed', default=2021, type=int)

    args = parser.parse_args(args)

    return args


def main(args):
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
    if args.data_name == 'NELL27K':
        args.save_path = './checkpoint/{}-{}-{}'.format(args.data_name, args.model, args.mode)
    else:
        args.save_path = './checkpoint/{}-{}-{}-{}'.format(args.data_name, args.model, args.noise_rate, args.mode)
    args.batch_size = config['batch_size']
    args.negative_sample_size = config['negative_sample_size']
    args.hidden_dim = config['hidden_dim']
    args.learning_rate = config['lr']
    args.gamma = config['gamma']
    args.adversarial_temperature = config['adversarial_temperature']
    args.max_steps = config['max_steps']
    args.update_steps = config['update_steps']
    args.ssvae_steps = config['ssvae_steps']
    if args.model == 'RotatE':
        args.double_entity_embedding = True

    if args.init_checkpoint:
        override_config(args)

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

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

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    true_train_triples = copy.deepcopy(train_triples)

    if args.data_name == 'NELL27K':
        file_name = 'noise.txt'
        noise_triples = np.loadtxt(os.path.join(args.data_path, file_name), dtype=np.int32)
        noise_triples = [tuple(x) for x in noise_triples.tolist()]
        train_triples = train_triples + noise_triples
    elif args.noise_rate is not 0:
        file_name = 'noise_' + str(args.noise_rate) + '.txt'
        noise_triples = np.loadtxt(os.path.join(args.data_path, file_name), dtype=np.int32)
        noise_triples = [tuple(x) for x in noise_triples.tolist()]
        train_triples = train_triples + noise_triples
    else:
        noise_triples = [(-1, -1, -1)]

    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))

    # All true triples
    all_true_triples = true_train_triples + valid_triples + test_triples

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        args=args,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
    )

    if args.vae_model == 'DGM':
        ssvae_model = DeepGenerativeModel([args.hidden_dim, 1, args.z_dim, eval(args.h_dim)])
    elif args.vae_model == 'ADM':
        ssvae_model = AuxiliaryDeepGenerativeModel([args.hidden_dim, 1, args.z_dim, args.a_dim, eval(args.h_dim)])

    logging.info('KGEModel Configuration:')
    logging.info(str(kge_model))
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.vae_model == 'DGM':
        logging.info('DeepGenerativeModel Configuration:')
    elif args.vae_model == 'ADM':
        logging.info('AuxiliaryDeepGenerativeModel Configuration:')
    logging.info(str(ssvae_model))
    logging.info('Model Parameter Configuration:')
    for name, param in ssvae_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    kge_model = kge_model.cuda()

    if args.mode == 'soft':
        ssvae_model = ssvae_model.cuda()

    if args.do_train:
        # Set training dataloader iterator for KGE.
        train_dataset_head = TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch')
        train_dataset_tail = TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch')
        train_dataloader_head = DataLoader(
            train_dataset_head,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=worker_init,
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            train_dataset_tail,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=worker_init,
            collate_fn=TrainDataset.collate_fn
        )
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        if args.mode == 'soft':
            # Set training dataloader iterator for labelled triples.
            labelled_triples = random.sample(train_triples, len(train_triples) // 10)
            labelled_dataset_head = TrainDataset(labelled_triples, nentity, nrelation,
                                                 1, 'head-batch')
            labelled_dataset_tail = TrainDataset(labelled_triples, nentity, nrelation,
                                                 1, 'tail-batch')
            labelled_dataset_head.true_head, labelled_dataset_head.true_tail = train_dataset_head.true_head, train_dataset_head.true_tail
            labelled_dataset_tail.true_head, labelled_dataset_tail.true_tail = train_dataset_tail.true_head, train_dataset_tail.true_tail
            labelled_dataset_head.count = train_dataset_head.count
            labelled_dataset_tail.count = train_dataset_tail.count
            labelled_dataloader_head = DataLoader(
                labelled_dataset_head,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                worker_init_fn=worker_init,
                collate_fn=TrainDataset.collate_fn
            )
            labelled_dataloader_tail = DataLoader(
                labelled_dataset_tail,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                worker_init_fn=worker_init,
                collate_fn=TrainDataset.collate_fn
            )
            labelled_iterator = BidirectionalOneShotIterator(labelled_dataloader_head, labelled_dataloader_tail)

            # Set training dataloader iterator for unlabelled triples.
            unlabelled_triples = list(set(train_triples) - set(labelled_triples))
            unlabelled_dataset_head = TrainDataset(unlabelled_triples, nentity, nrelation,
                                                   1, 'head-batch')
            unlabelled_dataset_tail = TrainDataset(unlabelled_triples, nentity, nrelation,
                                                   1, 'tail-batch')
            unlabelled_dataset_head.true_head, unlabelled_dataset_head.true_tail = train_dataset_head.true_head, train_dataset_head.true_tail
            unlabelled_dataset_tail.true_head, unlabelled_dataset_tail.true_tail = train_dataset_tail.true_head, train_dataset_tail.true_tail
            unlabelled_dataset_head.count = train_dataset_head.count
            unlabelled_dataset_tail.count = train_dataset_tail.count
            unlabelled_dataloader_head = DataLoader(
                unlabelled_dataset_head,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                worker_init_fn=worker_init,
                collate_fn=TrainDataset.collate_fn
            )
            unlabelled_dataloader_tail = DataLoader(
                unlabelled_dataset_tail,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                worker_init_fn=worker_init,
                collate_fn=TrainDataset.collate_fn
            )
            unlabelled_iterator = BidirectionalOneShotIterator(unlabelled_dataloader_head, unlabelled_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate,
            weight_decay=args.weight_decay,
        )

        if args.mode == 'soft':
            beta = DeterministicWarmup(n=2 * len(unlabelled_dataloader_head) * 100)
            sampler = ImportanceWeightedSampler(mc=args.mc, iw=args.iw)

            elbo = SVI(ssvae_model, likelihood=binary_cross_entropy, beta=beta, sampler=sampler)
            optimizerVAE = torch.optim.Adam(ssvae_model.parameters(), lr=3e-4, betas=(0.9, 0.999))

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        if 'score_weight' in kge_model.state_dict() and 'score_weight' not in checkpoint['model_state_dict']:
            checkpoint['model_state_dict']['score_weights'] = kge_model.state_dict()['score_weights']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            current_learning_rate = 0
    elif args.init_embedding:
        logging.info('Loading pretrained embedding %s ...' % args.init_embedding)
        if kge_model.entity_embedding is not None:
            entity_embedding = np.load(os.path.join(args.init_embedding, 'entity_embedding.npy'))
            relation_embedding = np.load(os.path.join(args.init_embedding, 'relation_embedding.npy'))
            entity_embedding = torch.from_numpy(entity_embedding).to(kge_model.entity_embedding.device)
            relation_embedding = torch.from_numpy(relation_embedding).to(kge_model.relation_embedding.device)
            kge_model.entity_embedding.data[:entity_embedding.size(0)] = entity_embedding
            kge_model.relation_embedding.data[:relation_embedding.size(0)] = relation_embedding
        init_step = 1
        current_learning_rate = 0
    else:
        logging.info('Randomly Initializing %s Model...' % args.model)
        init_step = 1

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('learning_rate = %.5f' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_sample_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    loss_func = nn.LogSigmoid()

    criterion = nn.BCELoss()

    if args.do_train:
        training_logs = []
        if args.mode == 'soft':
            soft = True
        rate = 10
        # Init confidence
        confidence = torch.ones(len(train_triples), requires_grad=False).cuda()
        # Training Loop
        for step in range(init_step, args.max_steps + 1):
            optimizer.zero_grad()
            log = kge_model.train_step(kge_model, train_iterator, confidence, loss_func, args)
            optimizer.step()

            training_logs.append(log)

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                }
                save_model(kge_model, ssvae_model, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, [metrics])
                training_logs = []

            if args.mode != 'none' and (step % args.update_steps == 0 or step == args.max_steps):
                torch.cuda.empty_cache()

                kge_model.eval()

                relation_embedding, entity_embedding = kge_model.get_embedding()

                kge_model.find_topk_triples_ssvae(kge_model, train_iterator, labelled_iterator, unlabelled_iterator, noise_triples, rate=rate)

                kge_model_func = kge_model.get_model_func()

                # Train ssvae.
                logging.info('Train ssvae...')
                alpha = args.loss_beta * (1 + len(unlabelled_dataloader_head) / len(labelled_dataloader_head))
                clf_loss = 0
                ssvae_model.train()
                for i in tqdm(range(args.ssvae_steps)):
                    pos, neg, sub_weight, mode, idx = next(labelled_iterator)
                    u_data, u_neg, sub_weight, unlabelled_mode, idx = next(unlabelled_iterator)
                    pos, neg = pos.cuda(), neg.cuda()
                    u_data = u_data.cuda()
                    batch_size, negative_sample = neg.size(0), neg.size(1)
                    h = torch.index_select(entity_embedding, 0, pos[:, 0])
                    r = torch.index_select(relation_embedding, 0, pos[:, 1])
                    t = torch.index_select(entity_embedding, 0, pos[:, 2])
                    pos_data = kge_model_func[kge_model.model_name](h, r, t, 'single', True).detach()
                    if mode == 'head-batch':
                        h = torch.index_select(entity_embedding, 0, neg.view(-1)).view(batch_size, negative_sample,
                                                                                       -1)
                        r = r.unsqueeze(1)
                        t = t.unsqueeze(1)
                    elif mode == 'tail-batch':
                        h = h.unsqueeze(1)
                        r = r.unsqueeze(1)
                        t = torch.index_select(entity_embedding, 0, neg.view(-1)).view(batch_size, negative_sample,
                                                                                       -1)
                    neg_data = kge_model_func[kge_model.model_name](h, r, t, 'single', True).detach()
                    neg_data = neg_data.view(batch_size, -1)
                    x = torch.cat([pos_data, neg_data], dim=0)
                    y = torch.cat([torch.ones(batch_size), torch.zeros(batch_size * negative_sample)], dim=0).view(-1, 1)

                    h = torch.index_select(entity_embedding, 0, u_data[:, 0])
                    r = torch.index_select(relation_embedding, 0, u_data[:, 1])
                    t = torch.index_select(entity_embedding, 0, u_data[:, 2])
                    u = kge_model_func[kge_model.model_name](h, r, t, 'single', True).detach()

                    x, y, u = x.cuda(), y.cuda(), u.cuda()

                    L = -elbo(x, y)
                    U = -elbo(u)
                    labels = ssvae_model.classify(x)
                    classification_loss = criterion(labels, y)
                    J_alpha = L + alpha * classification_loss + U
                    optimizerVAE.zero_grad()
                    J_alpha.backward()
                    optimizerVAE.step()
                    clf_loss += classification_loss.item()

                    if i % 200 == 0 and i != 0:
                        cur_log = {
                            'kge_step': step,
                            'ssvae_step': i,
                            'mean_classification_loss': clf_loss / 200,
                            'cur_loss': J_alpha.item()
                        }
                        logging.info(cur_log)
                        clf_loss = 0

                if step == args.max_steps:
                    # Detect error.
                    logging.info('Begin detect error...')
                    num_true = len(test_triples)
                    if args.data_name == 'NELL27K':
                        file_name = 'test_negative.txt'
                        test_negative_triples = np.loadtxt(os.path.join(args.data_path, file_name), dtype=np.int32)
                        test_negative_triples = [tuple(x) for x in test_negative_triples.tolist()]
                        all_test_triples = test_triples + test_negative_triples
                        error_triples = kge_model.error_detect(kge_model, ssvae_model, all_test_triples, num_true)
                        save_file_name = 'error_triples.txt'
                        np.savetxt(os.path.join(args.save_path, save_file_name), error_triples, fmt='%d', delimiter='\t')
                    else:
                        for error_rate in [10, 20, 30, 40, 50]:
                            logging.info('Error rate: {}'.format(error_rate))
                            file_name = 'test_negative_{}.txt'.format(str(error_rate))
                            test_negative_triples = np.loadtxt(os.path.join(args.data_path, file_name), dtype=np.int32)
                            test_negative_triples = [tuple(x) for x in test_negative_triples.tolist()]
                            all_test_triples = test_triples + test_negative_triples
                            error_triples = kge_model.error_detect(kge_model, ssvae_model, all_test_triples, num_true)
                            save_file_name = 'error_triples_{}.txt'.format(str(error_rate))
                            np.savetxt(os.path.join(args.save_path, save_file_name), error_triples, fmt='%d', delimiter='\t')

                else:
                    # Update confidence.
                    logging.info('Update confidence...')
                    kge_model.update_confidence(kge_model, ssvae_model, train_iterator, confidence, soft, len(true_train_triples), args)

                torch.cuda.empty_cache()

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
        }
        save_model(kge_model, ssvae_model, optimizer, save_variable_list, args)
    torch.cuda.empty_cache()

    if args.do_test:
        torch.cuda.empty_cache()
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)


if __name__ == '__main__':
    main(parse_args())

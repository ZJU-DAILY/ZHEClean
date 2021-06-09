import logging
import os

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader
from ote import OTE
from dataloader import TrainDataset, TestDataset
from helper import *

pi = 3.14159265358979323846


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, args,
                 double_entity_embedding=False, double_relation_embedding=False,
                 dropout=0, init_embedding=True):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0  # for embedding initialization
        self.scale_relation = True

        self.gamma = nn.Parameter(  # for embedding initialization and modeling (RotatE, TransE)
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.test_split_num = 1

        # (15 + 2) / 200 ~ 0.08
        # (9 + 2 ) / 1000 ~ 0.01
        self.embedding_range = nn.Parameter(
            torch.Tensor([0.01]),
            requires_grad=False
        )
        self._aux = {}

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim
        self.ote = None
        if model_name == 'OTE':
            assert self.entity_dim % args.ote_size == 0
            sub_emb_num = self.entity_dim // args.ote_size
            self.ote = OTE(args.ote_size, args.ote_scale)
            use_scale = self.ote.use_scale
            self.relation_dim = self.entity_dim * (args.ote_size + (1 if use_scale else 0))

            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding,
                a=0,
                b=1.0
            )
            if use_scale:
                self.relation_embedding.data.view(-1, args.ote_size + 1)[:,
                -1] = self.ote.scale_init()  # start with no scale
            # make initial relation embedding orthogonal
            rel_emb_data = self.orth_rel_embedding()
            self.relation_embedding.data.copy_(rel_emb_data.view(nrelation, self.relation_dim))
        else:
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if init_embedding:
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.uniform_(
                tensor=self.entity_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        else:
            self.entity_embedding = None

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        self.dropout = nn.Dropout(dropout) if dropout > 0 else lambda x: x

    def orth_rel_embedding(self):
        rel_emb_size = self.relation_embedding.size()
        ote_size = self.ote.num_elem
        scale_dim = 1 if self.ote.use_scale else 0
        rel_embedding = self.relation_embedding.view(-1, ote_size, ote_size + scale_dim)
        rel_embedding = self.ote.orth_embedding(rel_embedding).view(rel_emb_size)
        if rel_embedding is None:
            rel_embedding = self.ote.fix_embedding_rank(
                self.relation_embedding.view(-1, ote_size, ote_size + scale_dim))
            if self.training:
                self.relation_embedding.data.copy_(rel_embedding.view(rel_emb_size))
                rel_embedding = self.relation_embedding.view(-1, ote_size, ote_size + scale_dim)
            rel_embedding = self.ote.orth_embedding(rel_embedding, do_test=False).view(rel_emb_size)
        return rel_embedding

    def cal_embedding(self):
        if self.model_name == 'OTE':
            rel_embedding = self.orth_rel_embedding()
            self._aux['rel_emb'] = rel_embedding
            self._aux['ent_emb'] = self.entity_embedding

    def get_embedding(self):
        if self.model_name == 'OTE':
            return self._aux['rel_emb'], self._aux['ent_emb']
        return self.relation_embedding, self.entity_embedding

    def reset_embedding(self):
        for k in [key for key in self._aux.keys() if key != "static"]:
            self._aux[k] = None
        pass

    def get_model_func(self):
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'OTE': self.OTE,
        }
        return model_func

    def forward(self, sample, mode='single'):
        """
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        """

        relation_embedding, entity_embedding = self.get_embedding()
        if mode in ('single', 'head-single', 'tail-single'):
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
            head_ids = sample[:, 0].unsqueeze(1) if mode == 'head-single' else sample[:, 0]
            tail_ids = sample[:, 2].unsqueeze(1) if mode == 'tail-single' else sample[:, 2]
            self._aux['samples'] = (head_ids, sample[:, 1], tail_ids, mode)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)
            self._aux['samples'] = (head_part, tail_part[:, 1], tail_part[:, 2], mode)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            self._aux['samples'] = (head_part[:, 0], head_part[:, 1], tail_part, mode)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = self.get_model_func()
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode, topk=False):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        if topk:
            return score

        if mode == 'detect':
            score = - torch.norm(score, p=1, dim=1)
        else:
            score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode, topk=False):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        if topk:
            return score

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode, topk=False):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        if topk:
            return score

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode, topk=False):

        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation / (self.embedding_range.item() / pi) if self.scale_relation else relation

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score, im_score = RotatE_Trans((re_tail, im_tail), (re_relation, im_relation), False)
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score, im_score = RotatE_Trans((re_head, im_head), (re_relation, im_relation), True)
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        if topk:
            return score

        score = self.gamma.item() - score.sum(dim=-1)
        return score

    def OTE(self, head, relation, tail, mode, topk=False):
        if mode in ("head-batch", 'head-single'):
            relation = self.ote.orth_reverse_mat(relation)
            output = self.ote(tail, relation)
            if topk:
                return output - head
            score = self.ote.score(output - head)
        else:
            output = self.ote(head, relation)
            if topk:
                return output - tail
            score = self.ote.score(output - tail)
        score = self.gamma.item() - score

        return score

    @staticmethod
    def apply_loss_func(score, loss_func, is_negative_score=False, label_smoothing=0.1):
        if isinstance(loss_func, nn.SoftMarginLoss):
            tgt = -1 if is_negative_score else 1
            tgt = torch.empty(score.size()).fill_(tgt).to(score.device)
            output = loss_func(score, tgt)
        elif isinstance(loss_func, nn.BCELoss):
            # bceloss
            tgt = 0 if is_negative_score else 1
            if label_smoothing > 0:
                tgt = tgt * (1 - label_smoothing) + 0.0001
            tgt = torch.empty(score.size()).fill_(tgt).to(score.device)
            output = loss_func(score, tgt)
        else:
            output = loss_func(-score) if is_negative_score else loss_func(score)
        return output

    @staticmethod
    def train_step(model, train_iterator, confidence, loss_func, args, generator=None):
        """
        A single train step. Apply back-propation and return the loss.
        """
        model.train()

        positive_sample, negative_sample, subsampling_weight, mode, idxs = next(train_iterator)
        batch_confidence = torch.index_select(confidence, 0, torch.LongTensor(idxs).cuda())
        model.cal_embedding()

        positive_sample = positive_sample.cuda()
        if isinstance(negative_sample, tuple):
            negative_sample = [x.cuda() for x in negative_sample]
        else:
            negative_sample = negative_sample.cuda()
        # subsampling_weight = subsampling_weight.cuda()
        batch_confidence = batch_confidence.cuda()

        if generator is not None:
            positive_sample, negative_sample = generator.generate(model, positive_sample, negative_sample, mode,
                                                                  train=False, n_sample=args.negative_sample_size // 2)

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * model.apply_loss_func(negative_score, loss_func, True, args.label_smoothing)).sum(dim=1)
        else:
            negative_score = model.apply_loss_func(negative_score, loss_func, True, args.label_smoothing).mean(dim=1)
        pmode = "head-single" if mode == "head-batch" else "tail-single"
        positive_score = model(positive_sample, pmode)

        positive_score = model.apply_loss_func(positive_score, loss_func, False, args.label_smoothing).squeeze(dim=1)
        loss_sign = -1

        positive_sample_loss = loss_sign * (batch_confidence * positive_score).sum() / batch_confidence.sum()
        negative_sample_loss = loss_sign * (batch_confidence * negative_score).sum() / batch_confidence.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        model.reset_embedding()

        return log

    def split_test(self, sample, mode):
        if self.test_split_num == 1:
            return self(sample, mode)
        p_sample, n_sample = sample
        scores = []
        sub_samples = torch.chunk(n_sample, self.test_split_num, dim=1)
        for n_ss in sub_samples:
            scores.append(self((p_sample, n_ss.contiguous()), mode))
        return torch.cat(scores, dim=1)

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args, head_only=False, tail_only=False):
        """
        Evaluate the model on test dataset.
        """

        model.eval()
        model.cal_embedding()

        # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'head-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=4,
            worker_init_fn=worker_init,
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'tail-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=4,
            worker_init_fn=worker_init,
            collate_fn=TestDataset.collate_fn
        )
        if head_only:
            test_dataset_list = [test_dataloader_head]
        elif tail_only:
            test_dataset_list = [test_dataloader_tail]
        else:
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = [[] for i in test_dataset_list]

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for k, test_dataset in enumerate(test_dataset_list):
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score = model.split_test((positive_sample, negative_sample), mode)

                    score += filter_bias * (score.max() - score.min())

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = torch.nonzero(argsort[i, :] == positive_arg[i])
                        assert ranking.size(0) == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs[k].append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })
                        buf = "%s " % mode + "rank %d " % ranking + " ".join(("%s" % int(x) for x in
                                                                              positive_sample[
                                                                                  i])) + "\t"  # '[ %d %d %d ]\t'%(int(x) for x in positive_sample[i])
                        buf = buf + " ".join(["%d" % x for x in argsort[i][:10]])
                        logging.debug(buf)

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

            metrics = [{} for l in logs]
            for i, log in enumerate(logs):
                for metric in log[0].keys():
                    metrics[i][metric] = sum([lg[metric] for lg in log]) / len(log)
                if len(logs) > 1:
                    metrics[i]['name'] = "head-batch" if i == 0 else "tail-batch"
            if len(logs) == 2:
                metrics_all = {}
                log_all = logs[0] + logs[1]
                for metric in log_all[0].keys():
                    metrics_all[metric] = sum([lg[metric] for lg in log_all]) / len(log_all)
                metrics_all['name'] = "Overall"
                metrics.append(metrics_all)

        model.reset_embedding()
        return metrics

    @staticmethod
    def find_topk_triples_ssvae(model, train_iterator, labelled_iterator, unlabelled_iterator, noise_triples, rate=10):
        """
        Find topk triples.
        """
        model.eval()
        noise_triples = set(noise_triples)
        k = len(train_iterator.dataloader_head.dataset.triples) * rate // 100
        topk_heap = TopKHeap(k)
        all_triples = train_iterator.dataloader_head.dataset.triples
        model_func = model.get_model_func()
        relation_embedding, entity_embedding = model.get_embedding()
        i = 0
        while i < len(all_triples):
            j = min(i + 1024, len(all_triples))
            sample = torch.LongTensor(all_triples[i: j]).cuda()
            h = torch.index_select(entity_embedding, 0, sample[:, 0]).detach()
            r = torch.index_select(relation_embedding, 0, sample[:, 1]).detach()
            t = torch.index_select(entity_embedding, 0, sample[:, 2]).detach()
            s = model_func[model.model_name](h, r, t, 'single', True)
            score = (-torch.norm(s, p=1, dim=1)).view(-1).detach().cpu().tolist()
            for x, triple in enumerate(all_triples[i: j]):
                topk_heap.push((score[x], triple))
            i = j

        topk_list = topk_heap.topk()
        _, topk_triples = list(zip(*topk_list))
        labelled_iterator.dataloader_head.dataset.triples = topk_triples
        labelled_iterator.dataloader_tail.dataset.triples = [tri for tri in labelled_iterator.dataloader_head.dataset.triples]
        labelled_iterator.dataloader_head.dataset.len = len(topk_triples)
        labelled_iterator.dataloader_tail.dataset.len = len(topk_triples)

        num_fake = len(set(topk_triples).intersection(noise_triples))
        logging.info('Fake in top k triples %d / %d' % (num_fake, len(topk_triples)))

        unlabelled_triples = list(set(all_triples) - set(topk_triples))
        unlabelled_iterator.dataloader_head.dataset.triples = unlabelled_triples
        unlabelled_iterator.dataloader_tail.dataset.triples = [tri for tri in unlabelled_iterator.dataloader_head.dataset.triples]
        unlabelled_iterator.dataloader_head.dataset.len = len(unlabelled_triples)
        unlabelled_iterator.dataloader_tail.dataset.len = len(unlabelled_triples)

    @staticmethod
    def error_detect(kge_model, model, test_triples, true_num, use_sigmoid=False):
        """
        Detect error.
        """
        kge_model.eval()
        model.eval()
        kge_model_func = kge_model.get_model_func()
        relation_embedding, entity_embedding = kge_model.get_embedding()
        scores = []
        i = 0
        while i < len(test_triples):
            j = min(i + 1024, len(test_triples))
            sample = torch.LongTensor(test_triples[i: j]).cuda()
            h = torch.index_select(entity_embedding, 0, sample[:, 0]).detach()
            r = torch.index_select(relation_embedding, 0, sample[:, 1]).detach()
            t = torch.index_select(entity_embedding, 0, sample[:, 2]).detach()
            s = kge_model_func[kge_model.model_name](h, r, t, 'single', True).detach()
            s = s.view(sample.size(0), -1)
            c = model.classify(s)
            c = c.detach().view(-1)
            for score in c:
                scores.append(score.item())
            i = j

        error_triples = []
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

            TP = 0
            FN = 0
            FP = 0
            TN = 0

            for index, score in enumerate(scores):
                if index < true_num:
                    if score >= threshold:
                        TP += 1
                    else:
                        FN += 1
                        if threshold == 0.5:
                            error_triples.append(test_triples[index])
                else:
                    if score < threshold:
                        TN += 1
                        if threshold == 0.5:
                            error_triples.append(test_triples[index])
                    else:
                        FP += 1

            TN_acc = TN / (TN + FP)
            log = {
                'threshold': threshold,
                'TP': TP,
                'FN': FN,
                'FP': FP,
                'TN': TN,
                'TN-acc': TN_acc
            }
            logging.info(log)

        return error_triples

    @staticmethod
    def update_confidence(kge_model, model, train_iterator, confidence, soft, true_num, args):
        """
        Update confidence of all train triples.
        """
        kge_model.eval()
        model.eval()
        all_triples = train_iterator.dataloader_head.dataset.triples
        kge_model_func = kge_model.get_model_func()
        relation_embedding, entity_embedding = kge_model.get_embedding()
        threshold = args.threshold
        i = 0
        while i < len(all_triples):
            j = min(i + 1024, len(all_triples))
            sample = torch.LongTensor(all_triples[i: j]).cuda()
            h = torch.index_select(entity_embedding, 0, sample[:, 0]).detach()
            r = torch.index_select(relation_embedding, 0, sample[:, 1]).detach()
            t = torch.index_select(entity_embedding, 0, sample[:, 2]).detach()
            s = kge_model_func[kge_model.model_name](h, r, t, 'single', True).detach()
            s = s.view(sample.size(0), -1)
            c = model.classify(s)
            c = c.detach().view(-1)
            if soft:
                confidence[i: j] = c
            else:
                confidence[i: j] = (c >= threshold).type(torch.float32) + 0.00001
            i = j

        # TP = (confidence[0: true_num] >= threshold).cpu().sum().item()
        # FN = (confidence[0: true_num] < threshold).cpu().sum().item()
        # FP = (confidence[true_num:] >= threshold).cpu().sum().item()
        # TN = (confidence[true_num:] < threshold).cpu().sum().item()
        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        # F1 = 2 * precision * recall / (precision + recall)
        # acc = (TP + TN) / (TP + TN + FP + FN)
        #
        # log = {
        #     'threshold': threshold,
        #     'mean_pos_score': torch.mean(confidence[0: true_num]).item(),
        #     'mean_neg_score': torch.mean(confidence[true_num:]).item(),
        #     'TP': TP,
        #     'FN': FN,
        #     'FP': FP,
        #     'TN': TN,
        #     'precision': precision,
        #     'recall': recall,
        #     'F1': F1,
        #     'accuracy': acc
        # }
        # logging.info(log)

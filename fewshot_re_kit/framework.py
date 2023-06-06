from . import sentence_encoder
from . import data_loader

import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from datetime import datetime

import os
import numpy as np
import sys
import time
import pdb
import random

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

class FewShotREModel(nn.Module):
    def __init__(self, my_sentence_encoder,shots = None,entity_encoder = None):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(my_sentence_encoder)
        if entity_encoder is not None:
            self.entity_encoder = nn.DataParallel(entity_encoder)
        else:
            self.entity_encoder = None
        self.cost = nn.CrossEntropyLoss()
        self.shots = shots
    
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).to(torch.float))

class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, adv_data_loader=None, adv=False, d=None):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.adv_data_loader = adv_data_loader
        self.adv = adv
        if adv:
            self.adv_cost = nn.CrossEntropyLoss()
            self.d = d
            if torch.cuda.is_available():
                self.d.cuda()
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              B, N_for_train, N_for_eval, K, Q,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              bert_optim=False,
              pytorch_optim=optim.SGD,
              learning_rate=1e-1,
              lr_step_size=20000,
              weight_decay=1e-5,
              adv_dis_lr=1e-1,
              adv_enc_lr=1e-1,
              warmup_step=300,
              load_ckpt=None,
              save_ckpt=None,
              na_rate=0,
              grad_iter=1,
              fp16=False,
              pair=False,
              coef = 1.0
              ):
        print("Start training...")
    
        # Init
        if bert_optim:
            print('Use bert optim!')
            parameters_to_optimize = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize 
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

            '''
            unfreeze_layers = ['layer.11','bert.pooler','bert.classifier']
            for name, param in  model.named_parameters():
                print(name,param.size())
                param.requires_grad = False
                for ele in unfreeze_layers:
                    if ele in name:
                        param.requires_grad = True
                        break
            '''
            optimizer = pytorch_optim(parameters_to_optimize, lr = learning_rate)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_step, num_training_steps = train_iter)
            if self.adv:
                optimizer_encoder = pytorch_optim(parameters_to_optimize, lr = adv_enc_lr)
                optimizer_dis = pytorch_optim(self.d.parameters(), lr = adv_dis_lr)
        else:
            parameters_to_optimize = filter(lambda x: x.requires_grad, model.parameters())
            optimizer = pytorch_optim(parameters_to_optimize, learning_rate, weight_decay = weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = lr_step_size)
            if self.adv:
                optimizer_encoder = pytorch_optim(model.parameters(), lr = adv_enc_lr)
                optimizer_dis = pytorch_optim(self.d.parameters(), lr = adv_dis_lr)

        #count parameter
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0

        for param in model.parameters():
            mulValue = np.prod(param.size())
            Total_params += mulValue
            if param.requires_grad:
                Trainable_params += mulValue
            else:
                NonTrainable_params += mulValue

        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.train()
        if self.adv:
            self.d.train()

        # Training
        best_acc = 0
        iter_loss = 0.0
        iter_loss_dis = 0.0
        iter_right = 0.0
        iter_right_dis = 0.0
        iter_sample = 0.0
        for it in range(train_iter):
            begin = time.time()
            if pair:
                batch, label = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in batch:
                        batch[k] = batch[k].cuda()
                    label = label.cuda()
                logits, pred = model(batch, N_for_train, K, Q * N_for_train + na_rate * Q)
            else:
                support, query, label = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    label = label.cuda()
                logits, pred  = model(support, query, N_for_train, K, Q * N_for_train + na_rate * Q)

            loss = model.loss(logits, label) / float(grad_iter)
            right = model.accuracy(pred, label)/ float(grad_iter)
            
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 10)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Adv part
            if self.adv:
                support_adv = next(self.adv_data_loader)
                if torch.cuda.is_available():
                    for k in support_adv:
                        support_adv[k] = support_adv[k].cuda()

                # Generator Loss
                features_ori = model.sentence_encoder(support)
                features_adv = model.sentence_encoder(support_adv)
                if model.entity_encoder is not None:
                    entity_ori = model.entity_encoder(support)
                    features_ori = torch.cat([features_ori, entity_ori], dim=-1)
                    entity_adv = model.entity_encoder(support_adv)
                    features_adv = torch.cat([features_adv, entity_adv], dim=-1)

                features = torch.cat([features_ori, features_adv], 0)
                total = features.size(0)
                if torch.cuda.is_available():
                    dis_labels = torch.cat([torch.zeros((total//2)).long().cuda(),
                        torch.ones((total//2)).long().cuda()], 0)
                else:
                    dis_labels = torch.cat([torch.zeros((total//2)).long(),
                        torch.ones((total//2)).long()], 0)

                dis_logits = self.d(features)
                loss_encoder = self.adv_cost(dis_logits, 1 - dis_labels)
                loss_encoder.backward()
                optimizer_encoder.step()
                optimizer_dis.zero_grad()
                optimizer_encoder.zero_grad()

                # Discriminator Loss
                features_ori = model.sentence_encoder(support).detach()
                features_adv = model.sentence_encoder(support_adv).detach()
                if model.entity_encoder is not None:
                    entity_ori = model.entity_encoder(support)
                    features_ori = torch.cat([features_ori, entity_ori], dim=-1)
                    entity_adv = model.entity_encoder(support_adv)
                    features_adv = torch.cat([features_adv, entity_adv], dim=-1)
                features = torch.cat([features_ori, features_adv], 0)
                total = features.size(0)
                if torch.cuda.is_available():
                    dis_labels = torch.cat([torch.zeros((total // 2)).long().cuda(),
                        torch.ones((total // 2)).long().cuda()], 0)
                else:
                    dis_labels = torch.cat([torch.zeros((total//2)).long(),
                        torch.ones((total//2)).long()], 0)

                dis_logits = self.d(features)
                loss_dis = coef * self.adv_cost(dis_logits, dis_labels)
                _, pred = dis_logits.max(-1)
                right_dis = float((pred == dis_labels).long().sum()) / float(total)

                loss_dis.backward()
                optimizer_dis.step()
                optimizer_dis.zero_grad()
                optimizer_encoder.zero_grad()

                iter_loss_dis += self.item(loss_dis.data)
                iter_right_dis += right_dis
            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1

            log_iter_time = time.time() - begin #batch time
            log_remain_time_s = log_iter_time * ((train_iter - it) + np.ceil((train_iter - it) / val_step) * val_iter)
            log_remain_time_h = log_remain_time_s / 3600

            if self.adv:
                sys.stdout.write('step: {0:4} | loss: {1:2.4f}, accuracy: {2:3.2f}%, dis_loss: {3:2.4f}, dis_acc: {4:2.2f}%, iter time: {5:.4f}s, remain time: {6:.2f}s ({7:.2f}h)'.format(
                        it + 1,
                        iter_loss / iter_sample, 
                        100 * iter_right / iter_sample,
                        iter_loss_dis / iter_sample,
                        100 * iter_right_dis / iter_sample,
                        log_iter_time,
                        log_remain_time_s,
                        log_remain_time_h
                        )
                    + '\r')
            else:
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%, iter time: {3:.4f}s, remain time: {4:.2f}s ({5:.2f}h)'.format(
                            it + 1,
                            iter_loss / iter_sample,
                            100 * iter_right / iter_sample,
                            log_iter_time,
                            log_remain_time_s,
                            log_remain_time_h
                            )
                        +'\r')
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                acc = self.eval(model, B, N_for_eval, K, Q, val_iter, na_rate=na_rate, pair=pair)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    print(f"Saved model in {save_ckpt}")
                    best_acc = acc

                iter_loss = 0.
                iter_loss_dis = 0.
                iter_right = 0.
                iter_right_dis = 0.
                iter_sample = 0.
                
        print("\n####################\n")
        print("Finish training " + model_name)


    def eval(self,
         model,
         B, N, K, Q,
         eval_iter,
         na_rate=0,
         pair=False,
         ckpt=None, test_online=False):
        print("")

        model.eval()
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            state_dict = self.__load_model__(ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        with torch.no_grad():
            result = []
            for it in range(eval_iter):
                #begin = time.time()
                if test_online is True:
                    if pair:
                        batch = next(eval_dataset)
                        if torch.cuda.is_available():
                            for k in batch:
                                batch[k] = batch[k].cuda()
                        logits, pred = model(batch, N, K, Q * N + Q * na_rate)
                        result.extend(pred.cpu().numpy().tolist())
                        if it == eval_iter - 1:
                            return result
                    else:
                        support, query = next(eval_dataset)
                        if torch.cuda.is_available():
                            for k in support:
                                support[k] = support[k].cuda()
                            for k in query:
                                query[k] = query[k].cuda()
                        logits, pred = model(support, query, N, K, Q * 1 + Q * na_rate)
                        result.extend(pred.cpu().numpy().tolist())
                        if it == eval_iter - 1:
                            return result
                elif pair:
                    batch, label = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in batch:
                            batch[k] = batch[k].cuda()
                        label = label.cuda()
                    logits, pred = model(batch, N, K, Q * N + Q * na_rate)

                    right = model.accuracy(pred, label)
                    iter_right += self.item(right.data)
                    iter_sample += 1

                    sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1,100 * iter_right / iter_sample) + '\r')
                    sys.stdout.flush()
                else:
                    support, query, label = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in support:
                            support[k] = support[k].cuda()
                        for k in query:
                            query[k] = query[k].cuda()
                        label = label.cuda()
                    logits, pred = model(support, query, N, K, Q * N + Q * na_rate)

                    right = model.accuracy(pred, label)
                    iter_right += self.item(right.data)
                    iter_sample += 1

                    sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) +'\r')
                    sys.stdout.flush()
                #end = time.time()
                #print(f"{N}-way-{K}-shot instance test time : {(end - begin):.2f}s")
            print("")
            return iter_right / iter_sample

    def cluster(self, model, ckpt, unlabel_dataset, n_clusters, pseudo_pth, feature_pth, M, origin_unsupervised_path, origin_train_path, root_path):
        def softmax(x, axis=None):
            x = x - x.max(axis=axis, keepdims=True)
            y = np.exp(x)
            return y / y.sum(axis=axis, keepdims=True)

        def cal_dis(sample, center):
            result = []
            for i in center:
                result.append(np.sum((sample - i) ** 2))
            return np.array(result)

        def cal_entropy(dis):
            h = softmax(-dis)
            # print('dis',dis)
            h = np.multiply(h, -np.log(h))
            # print('h',h)
            result = h.sum(axis=-1, keepdims=False)
            # print('result',result)
            return result

        model.eval()
        if ckpt is not None:
            state_dict = self.__load_model__(ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)

        import numpy as np
        from sklearn.cluster import KMeans
        import json

        with torch.no_grad():
            samples = next(unlabel_dataset)
            features = model.sentence_encoder(samples)
            if model.entity_encoder is not None:
                entity = model.entity_encoder(samples)
                features = torch.cat([features, entity], dim=-1)
            features = features.cpu().detach().numpy()
        np.save("./data/" + feature_pth + ".npy", features)

        # cluster
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", copy_x=True)
        kmeans.fit(features)
        label = kmeans.labels_
        center = kmeans.cluster_centers_
        raw_data = json.load(open(origin_unsupervised_path))

        #select
        if M!= 0:
            entropy = []
            for sample in features:
                dis = cal_dis(sample, center)
                entropy.append(cal_entropy(dis))
            entropy_sort = np.sort(entropy)
            thresh = entropy_sort[len(features) // M]
            cluster_dict = {}
            for i, j, each_entropy in zip(label,raw_data,entropy):
                if each_entropy <= thresh:
                    if str(i) not in cluster_dict:
                        cluster_dict[str(i)] = [j]
                    else:
                        cluster_dict[str(i)].append(j)
        else:
            cluster_dict = {}
            for i in label:
                cluster_dict[str(i)] = []
            for i, j in zip(label, raw_data):
                cluster_dict[str(i)].append(j)

        train_wiki_dict = json.load(open(origin_train_path))
        train_wiki_dict.update(cluster_dict)
        if 'parse' in root_path:
            pseudo_path = os.path.join(root_path, pseudo_pth+ "_parse.json")
        else:
            pseudo_path = os.path.join(root_path, pseudo_pth+ ".json")
        with open(pseudo_path, "w") as f:
            json.dump(train_wiki_dict, f)
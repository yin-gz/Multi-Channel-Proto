import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class Multi(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, entity_encoder = None, hidden_size = 240, shots = 5, hybird_attention = True, drop_rate = 0.2):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder,shots,entity_encoder)
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(drop_rate)

        # for instance-level attention
        self.fc = nn.Linear(hidden_size, hidden_size, bias=True)
        # for feature-level attention
        self.conv1 = nn.Conv2d(1, 16, (shots, 1), padding=(shots // 2, 0))
        self.conv_final = nn.Conv2d(16, 1, (shots, 1), stride=(shots, 1))
        self.hybird_attention = hybird_attention
        self.conv_c1 = nn.Conv2d(1, 1, (shots, self.hidden_size//3),stride=(shots, self.hidden_size//3))

    def __dist__(self, x, y, dim):
        return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def hatt__dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)

    def hatt__batch_dist__(self, S, Q, score=None):
        return self.hatt__dist__(S, Q.unsqueeze(2), 3, score)

    def forward(self, support, query, N, K, total_Q):
        support_context = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query_context = self.sentence_encoder(query)  # (B * total_Q, D)

        support_entity = self.entity_encoder(support)
        query_entity = self.entity_encoder(query)
        support = torch.cat([support_context, support_entity],dim = -1)
        query = torch.cat([query_context, query_entity], dim= -1)

        hidden_size = support.size(-1)

        if self.hybird_attention is False:
        #concat
            support = self.drop(support)
            query = self.drop(query)
            support = support.view(-1, N, K, hidden_size)  # (B, N, K, D)
            query = query.view(-1, total_Q, hidden_size)  # (B, total_Q, D)
            support = torch.mean(support, 2)  # Calculate prototype for each class，(B, N, D)
            logits = self.__batch_dist__(support, query)  # (B, total_Q, N)
            logits = logits.view(-1, N)# (B*total_Q, N)
            _, pred = torch.max(logits.view(-1, N), 1)

        else:
            support = support.view(-1, N, K, hidden_size)  # (B, N, K, D)
            query = query.view(-1, total_Q, hidden_size)  # (B, N * Q, D)
            B = support.size()[0]
            NQ = query.size(1)  # Num of instances for each batch in the query set

            # feature-level attention
            fea_att_support= support.view(B * N, 1, K, self.hidden_size)  # (B * N, 1, K, D)
            if K == 1:
                #expand to train_K(5)
                fea_att_support = fea_att_support.expand(B * N, 1, 5, self.hidden_size)
            elif K != 5:
                #sample to train_K(5)
                idx = torch.tensor(np.random.choice(fea_att_support.size()[2], 5),dtype=torch.long)
                fea_att_support = fea_att_support.permute(2,1,0,3)[idx].permute(2,1,0,3)

            fea_att_score = F.relu(self.conv1(fea_att_support))  # (B * N, 16, K, D)
            fea_att_score = self.drop(fea_att_score)
            fea_att_score = self.conv_final(fea_att_score)  # (B * N, 1, 1, D)
            fea_att_score = F.relu(fea_att_score)
            fea_att_score = fea_att_score.view(B, N, self.hidden_size).unsqueeze(1)  # (B, 1, N, D)

            # instance-level attention
            support = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1)  # (B, NQ, N, K, D)
            support_for_att = self.fc(support) # (B, NQ, N, K, D)
            query_for_att = self.fc(query.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, K, -1))# (B, NQ, N, K, D)
            ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1)
            support_proto = (support * ins_att_score.unsqueeze(4).expand(-1, -1, -1, -1, self.hidden_size)).sum(3)  # (B, NQ, N, D)

            # Prototypical Networks
            logits = -self.hatt__batch_dist__(support_proto, query ,fea_att_score)
            _, pred = torch.max(logits.view(-1, N), 1)

        return logits, pred
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


POS_TO_ID = {'NNP': 1, 'NN': 2, 'CD': 3, 'VBD': 4, 'IN': 5, 'JJ': 6, 'VBN': 7, 'NNPS': 8, 'RB': 9, 'TO': 10, 'VB': 11, 'WDT': 12, 'VBZ': 13, 'RBS': 14, 'PRP': 15, 'HYPH': 16, None: 17, 'NNS': 18, 'VBP': 19, 'CC': 20, 'PRP$': 21, 'JJS': 22, 'POS': 23, 'WRB': 24, 'DT': 25, 'MD': 26, 'SYM': 27, 'EX': 28, 'VBG': 29, 'RP': 30, 'JJR': 31, 'FW': 32, 'RBR': 33, 'AFX': 34, 'WP': 35, 'NFP': 36, 'WP$': 37, 'UH': 38, '$': 39, 'PDT': 40, 'GW': 41, 'LS': 42}
DEP_TO_ID = {'compound': 0, 'pcomp': 1, 'case': 2, 'auxpass': 3, 'ccomp': 4, 'acomp': 5, 'punct': 6, 'xcomp': 7, 'poss': 8, 'dep': 9, 'nn': 10, 'partmod': 11, 'cop': 12, 'nmod': 13, 'pobj': 14, 'tmod': 15, 'amod': 16, 'number': 17, 'quantmod': 18, 'det': 19, 'mark': 20, 'csubj': 21, 'rcmod': 22, 'possessive': 23, 'fixed': 24, 'parataxis': 25, 'advmod': 26, 'mwe': 27, 'iobj': 28, 'num': 29, 'nsubj': 30, 'nsubjpass': 31, 'prep': 32, 'goeswith': 33, 'preconj': 34, 'dobj': 35, 'cc': 36, 'appos': 37, 'prt': 38, 'aux': 39, 'obj': 40, 'acl': 41, 'discourse': 42, 'infmod': 43, 'root': 44, 'obl': 45, 'npadvmod': 46, 'nummod': 47, 'conj': 48, 'expl': 49, 'neg': 50, 'advcl': 51, 'None': 52, 'predet': 53, 'csubjpass': 54}

class Embedding(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5,parse = True,mask = False):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.mask_control = mask
        self.parse = parse
        
        # Word embedding
        word_vec_mat = torch.from_numpy(word_vec_mat)#word_vec_MAT是numpy矩阵，是按词表顺序排列的
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0], self.word_embedding_dim, padding_idx=word_vec_mat.shape[0] - 1)
        self.word_embedding.weight.data.copy_(word_vec_mat)

        # Position Embedding
        self.pos_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)

        # Dis Embedding
        self.dis_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx=0)

        # DEPREL Embedding
        self.dep_embedding = nn.Embedding(len(DEP_TO_ID)+1, pos_embedding_dim, padding_idx=0)

        # PART of SPEECH Embedding
        self.ps_embedding = nn.Embedding(len(POS_TO_ID)+1, pos_embedding_dim, padding_idx=0)

    def forward(self, inputs):
        word = inputs['word'] #[B,length]
        pos1 = inputs['pos1'] #[B,length]
        pos2 = inputs['pos2'] #[B,length]

        if self.parse is True:
            dep1 = inputs['subj_deprel']
            dep2 = inputs['obj_deprel']
            dis1 = inputs['subj_dis']
            dis2 = inputs['obj_dis']
            #POS = inputs['POS']

        #get subj_word and obj_word
        mask_subj = pos1.eq(self.max_length).eq(0)
        subj_word = word.masked_fill(mask_subj, 0)
        mask_obj = pos2.eq(self.max_length).eq(0)
        obj_word = word.masked_fill(mask_obj, 0)

        if self.mask_control is True:
            # entity mask for the context(only in multi)
            mask_entity = torch.mul(pos1-self.max_length, pos2-self.max_length)
            mask_entity = mask_entity.eq(0)
            word = word.masked_fill(mask_entity, 0)
            pos1 = pos1.masked_fill(mask_entity, 0)
            pos2 = pos2.masked_fill(mask_entity, 0)
            if self.parse is True:
                dep1 = dep1.masked_fill(mask_entity, 0)
                dep2 = dep2.masked_fill(mask_entity, 0)
                dis1 = dis1.masked_fill(mask_entity, 0)
                dis2 = dis2.masked_fill(mask_entity, 0)
                #POS = POS.masked_fill(mask_entity, 0)

        if self.parse is True:
            aff_info = torch.cat([self.dis_embedding(dis1), self.dis_embedding(dis2), self.dep_embedding(dep1), self.dep_embedding(dep2),self.pos_embedding(pos1),self.pos_embedding(pos2)], 2)
        else:
            aff_info = torch.cat([self.pos_embedding(pos1), self.pos_embedding(pos2)], 2)

        x = self.word_embedding(word)
        subj_word = self.word_embedding(subj_word)
        obj_word = self.word_embedding(obj_word)

        return x, aff_info, subj_word, obj_word



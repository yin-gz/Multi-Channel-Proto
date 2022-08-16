import torch
import torch.nn as nn
import math
import numpy as np
from . import network
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from .utils import *
from torch.nn import functional as F

DEP_TO_ID = {'compound': 0, 'pcomp': 1, 'case': 2, 'auxpass': 3, 'ccomp': 4, 'acomp': 5, 'punct': 6, 'xcomp': 7, 'poss': 8, 'dep': 9, 'nn': 10, 'partmod': 11, 'cop': 12, 'nmod': 13, 'pobj': 14, 'tmod': 15, 'amod': 16, 'number': 17, 'quantmod': 18, 'det': 19, 'mark': 20, 'csubj': 21, 'rcmod': 22, 'possessive': 23, 'fixed': 24, 'parataxis': 25, 'advmod': 26, 'mwe': 27, 'iobj': 28, 'num': 29, 'nsubj': 30, 'nsubjpass': 31, 'prep': 32, 'goeswith': 33, 'preconj': 34, 'dobj': 35, 'cc': 36, 'appos': 37, 'prt': 38, 'aux': 39, 'obj': 40, 'acl': 41, 'discourse': 42, 'infmod': 43, 'root': 44, 'obl': 45, 'npadvmod': 46, 'nummod': 47, 'conj': 48, 'expl': 49, 'neg': 50, 'advcl': 51, 'None': 52, 'predet': 53, 'csubjpass': 54}

class CNNSentenceEncoder(nn.Module):
    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50,
            pos_embedding_dim=5, hidden_size=240, mode = None, mask = False, parse = False, word_att = None):

        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, 
                word_embedding_dim, pos_embedding_dim, parse, mask)
        self.mode = mode
        self.word_att = word_att
        aff_info_dim = 6 * pos_embedding_dim

        if self.hidden_size != 0:
            self.context_conv = nn.Conv1d(word_embedding_dim, self.hidden_size, kernel_size = 3, padding = 1)
        else:
            self.context_conv = nn.Conv1d(word_embedding_dim, 1, kernel_size=3, padding=1)

        if parse is True:
            if self.word_att is False:
                self.encoder = network.encoder.Encoder(max_length, word_embedding_dim + aff_info_dim, pos_embedding_dim, hidden_size)
            else:
                self.att_conv = nn.Conv1d(aff_info_dim, 4, kernel_size=3, padding=1)
        else:#only word_embedding_dim + 2*pos_emb
            self.encoder = network.encoder.Encoder(max_length, word_embedding_dim + 2* pos_embedding_dim,
                                                    pos_embedding_dim, hidden_size)

        #Encoder for head entity
        self.encoder_sub = network.encoder.Encoder(max_length, word_embedding_dim, 0, hidden_size)
        #Encoder for tail entity
        self.encoder_obj = network.encoder.Encoder(max_length, word_embedding_dim, 0, hidden_size)
        self.word2id = word2id


    def forward(self, inputs):
        x, aff_info, subj_word, obj_word = self.embedding(inputs)

        if self.mode == 'context' or self.mode is None:
            if self.word_att is False:
                x = self.encoder(torch.cat([x,aff_info],2))
            else:
                zero_index = torch.where(aff_info !=0, torch.ones(1).cuda(), torch.zeros(1).cuda())
                zero_index = torch.index_select(zero_index,dim=-1,index = torch.tensor([0]).cuda())
                aff_info = aff_info.transpose(1, 2)
                att_score = self.att_conv(aff_info).transpose(1, 2)
                #set PADDING ATT to zero
                att_score = att_score*zero_index
                att_score = torch.sum(att_score,dim=-1,keepdim = True)#(B,length,1)
                att_score = F.softmax(torch.where(att_score != 0, att_score, -9e10 * torch.ones_like(att_score)), dim=1)#(B,length,1)
                x = self.context_conv(x.transpose(1, 2))#(B,hidden,length)
                x = x.transpose(1, 2) * att_score#(B,length,hidden)
                x = torch.sum(x,dim = 1)
            return x

        elif self.mode == 'entity':
            subj_word = self.encoder_sub(subj_word)
            obj_word = self.encoder_obj(obj_word)
            return torch.cat([subj_word, obj_word], 1)

    def tokenize(self, raw_tokens, pos_head, pos_tail, subj_deprel = None, obj_deprel = None, subj_dis = None, obj_dis = None, POS = None):
        # token -> index
        indexed_tokens = []
        for index,token in enumerate(raw_tokens):
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[PAD]'])

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        # position
        pos1 = np.zeros((self.max_length), dtype=np.int32)#PADDING as 0
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        pos_head = pos_head[-1][0]
        pos_tail = pos_tail[-1][0]
        pos1_in_index = min(self.max_length, pos_head[0])
        pos2_in_index = min(self.max_length, pos_tail[0])
        count_pos = min(len(raw_tokens),self.max_length)
        for i in range(count_pos):
            if i in range(pos1_in_index,pos_head[-1]+1):
                pos1[i] = self.max_length
            else:
                pos1[i] = i - pos1_in_index + self.max_length
            if i in range(pos2_in_index,pos_tail[-1]+1):
                pos2[i] = self.max_length
            else:
                pos2[i] = i - pos2_in_index + self.max_length

        # mask
        no_padding_length = min(self.max_length, len(raw_tokens))
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:no_padding_length] = 1

        if subj_deprel is None:
            return indexed_tokens, pos1, pos2, mask

        no_padding_length = min(self.max_length, len(obj_dis), len(subj_dis), len(subj_deprel), len(obj_deprel))

        #deprel
        raw_s_deprel = subj_deprel
        raw_o_deprel = obj_deprel
        subj_deprel = np.zeros((self.max_length), dtype=np.int32)
        obj_deprel = np.zeros((self.max_length), dtype=np.int32)
        subj_deprel[:no_padding_length] = [DEP_TO_ID[item] for item in raw_s_deprel[:no_padding_length]]
        obj_deprel[:no_padding_length] = [DEP_TO_ID[item] for item in raw_o_deprel[:no_padding_length]]

        #distance
        raw_s_dis = np.array([(i+self.max_length) for i in subj_dis])
        raw_o_dis = np.array([(i+self.max_length) for i in obj_dis])
        subj_dis = np.zeros((self.max_length), dtype=np.int32)
        obj_dis = np.zeros((self.max_length), dtype=np.int32)
        subj_dis[:no_padding_length] = raw_s_dis[:no_padding_length]
        obj_dis[:no_padding_length] = raw_o_dis[:no_padding_length]

        return indexed_tokens, pos1, pos2, mask, subj_deprel, obj_deprel, subj_dis, obj_dis


class BERTSentenceEncoder(nn.Module):
    def __init__(self, pretrain_path, max_length):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)

    def forward(self, inputs):
        _, x = self.bert(inputs['word'], attention_mask=inputs['mask'])
        return x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)

        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, pos1, pos2, mask

class BERTPAIRSentenceEncoder(nn.Module):
    def __init__(self, pretrain_path, max_length):
        nn.Module.__init__(self)
        self.bert = BertForSequenceClassification.from_pretrained(
            pretrain_path,
            num_labels=2)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path + "/vocab.txt")

    def forward(self, inputs):
        x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[0]
        return x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        cur_pos = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        return indexed_tokens

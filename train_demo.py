from fewshot_re_kit.data_loader import get_loader, get_loader_pair, get_loader_unsupervised, get_loader_all_unsupervised,get_loader_test,get_loader_test_pair
from fewshot_re_kit.data_loader_offline_parse import get_loader_parse, get_loader_unsupervised_parse, get_loader_all_unsupervised_parse, get_loader_test_parse
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder
from models.proto import Proto
from models.pair import Pair
from models.proto_hatt import ProtoHATT
from models.multi import Multi
from models.d import Discriminator

import torch
from torch import optim, nn
import numpy as np

import json
import argparse
import os
import time


#default hyper parameters
class HP:
    # datasets
    train_set = "train_wiki_and_pseudo_pubmed"
    val_set = "val_pubmed"
    test_set = "val_pubmed"
    adv_set = "unsupervised_pubmed"
    #adv_set = None

    # N-way K-shot settings
    batch_size = 4
    trainN = 5
    N = 5
    K = 5
    Q = 5
    
    # Model params
    model = "multi"
    encoder = "cnn"
    hidden_size = 240
    pair = False
    dropout = 0.2
    max_length = 128

    # Multi channel encoder
    hybird_attention = True
    entity_size = 60  # context_size = hidden_size - 2*entity_size
    parse = True #parse or not
    parse_path = 'origin_parse'
    word_emb = 50
    pos_emb = 2
    word_att = True

    # Joint learning
    n_clusters = 10
    cluster = True
    pseudo_pth = "train_wiki_and_pseudo_pubmed"
    feature_pth = "unlabel_features"
    M = 2  # select top 1/M, 0 denotes no selecting


    # Training params
    train_iter = 5000
    val_iter = 1000
    val_step = 1000
    test_iter = 5000
    optim = "sgd"
    lr = 1e-1
    lr_step_size = 4000
    weight_decay = 1e-5
    adv_dis_lr = 1e-1
    adv_enc_lr = 1e-1
    warmup_step = 300

    # Save and load
    load_ckpt = None
    save_ckpt = None
    only_test = False

    # Test on the official website
    test_online = 'test_pubmed_input'

    # Others
    fp16 = False
    grad_iter = 1
    na_rate = 0
    

def main():
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument('--train', default=HP.train_set,
            help='train file')
    arg_parser.add_argument('--val', default=HP.val_set,
            help='val file')
    arg_parser.add_argument('--test', default=HP.test_set,
            help='test file')
    arg_parser.add_argument('--adv', default=HP.adv_set,
            help='adv file')
   
    arg_parser.add_argument('--batch_size', default=HP.batch_size, type=int,
            help='batch size')
    arg_parser.add_argument('--trainN', default=HP.trainN, type=int,
            help='N in training phase')
    arg_parser.add_argument('--N', default=HP.N, type=int,
            help='N way')
    arg_parser.add_argument('--K', default=HP.K, type=int,
            help='K shot')
    arg_parser.add_argument('--Q', default=HP.Q, type=int,
            help='Num of query per class')
    
    arg_parser.add_argument('--model', default=HP.model,
            help='model name')
    arg_parser.add_argument('--encoder', default=HP.encoder,
            help='encoder: cnn or bert')
    arg_parser.add_argument('--pair', default=HP.pair,
            help='use pair model')
    arg_parser.add_argument('--hidden_size', default=HP.hidden_size, type=int,
            help='hidden size')
    arg_parser.add_argument('--dropout', default=HP.dropout, type=float,
            help='dropout rate')
    arg_parser.add_argument('--max_length', default=HP.max_length, type=int,
            help='max length')

    # Multi channel encoder
    arg_parser.add_argument('--hybird_attention', default=HP.hybird_attention,
            help='hybird attention')
    arg_parser.add_argument('--entity_size', default=HP.entity_size, type=int,
            help='entity_size')
    arg_parser.add_argument('--parse', default=HP.parse,
            help='parse the context or not')
    arg_parser.add_argument('--parse_path', default=HP.parse_path,
            help='parse file path')
    arg_parser.add_argument('--word_emb', default=HP.word_emb, type=int,
            help='hybird attention')
    arg_parser.add_argument('--pos_emb', default=HP.pos_emb, type=int,
            help='pos_size')
    arg_parser.add_argument('--word_att', default=HP.word_att,
            help='word-level attention based on the dependency tree')

    # Joint learning
    arg_parser.add_argument('--n_clusters', default=HP.n_clusters, type=int,
           help='num of clusters')
    arg_parser.add_argument('--cluster', default=HP.cluster, action="store_true",
           help='cluster')
    arg_parser.add_argument('--pseudo_pth', default=HP.pseudo_pth)
    arg_parser.add_argument('--feature_pth', default=HP.feature_pth)
    # params for selecting pseudo labels
    arg_parser.add_argument('--M', default=HP.M, help='select top 1/M, 0 denotes no selecting')

    # Training params
    arg_parser.add_argument('--train_iter', default=HP.train_iter, type=int,
            help='num of iters in training')
    arg_parser.add_argument('--val_iter', default=HP.val_iter, type=int,
            help='num of iters in validation')
    arg_parser.add_argument('--test_iter', default=HP.test_iter, type=int,
            help='num of iters in testing')
    arg_parser.add_argument('--val_step', default=HP.val_step, type=int,
            help='val after training how many iters')
    arg_parser.add_argument('--optim', default=HP.optim,
            help='sgd / adam / adamw')
    arg_parser.add_argument('--lr', default=HP.lr, type=float,
            help='learning rate')
    arg_parser.add_argument('--lr_step_size', default=HP.lr_step_size, type=int,
            help='learning rate step')
    arg_parser.add_argument('--weight_decay', default=HP.weight_decay, type=float,
            help='weight decay')
    arg_parser.add_argument('--adv_dis_lr', default=HP.adv_dis_lr, type=float,
            help='adv dis lr')
    arg_parser.add_argument('--adv_enc_lr', default=HP.adv_enc_lr, type=float,
            help='adv enc lr')
    arg_parser.add_argument('--warmup_step', default=HP.warmup_step, type=int,
            help='warmup step')

    # Load
    arg_parser.add_argument('--load_ckpt', default=HP.load_ckpt,
            help='load ckpt')
    arg_parser.add_argument('--save_ckpt', default=HP.save_ckpt,
            help='save ckpt')
    arg_parser.add_argument('--only_test', action="store_true",default=HP.only_test,
            help='only test')

    # test on the official website
    arg_parser.add_argument('--test_online', default=HP.test_online,
            help='online test')


    # others
    arg_parser.add_argument('--grad_iter', default=HP.grad_iter, type=int,
            help='accumulate gradient every x iterations')
    arg_parser.add_argument('--fp16', default=HP.fp16,
            help='use nvidia apex fp16')
    arg_parser.add_argument('--na_rate', default=HP.na_rate,
            help='na_rate')

    opt = arg_parser.parse_args()
    print()
    print(opt)
    print()

    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length
    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))
    
    if encoder_name == 'cnn':
        try:
            glove_mat = np.load('./pretrain/glove/glove_mat.npy',allow_pickle=True)
            print('success load glove_mat!')
            glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
            print('success load glove_word2id!')
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        
        sentence_encoder = CNNSentenceEncoder(
                glove_mat,
                glove_word2id,
                max_length,
                hidden_size=opt.hidden_size,
                mask = False, #mask entity or not
                parse = opt.parse,
                word_embedding_dim = opt.word_emb,
                pos_embedding_dim = opt.pos_emb,
                word_att = opt.word_att
                )
    elif encoder_name == 'bert':
        if opt.pair:
            sentence_encoder = BERTPAIRSentenceEncoder(
                    './pretrain/bert-base-uncased',
                    max_length)
        else:
            sentence_encoder = BERTSentenceEncoder(
                    './pretrain/bert-base-uncased',
                    max_length)
    else:
        raise NotImplementedError
    
    if opt.pair:
        root_path = './data/origin/train'
        train_data_loader = get_loader_pair(opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size,root=root_path)
        val_data_loader = get_loader_pair(opt.val, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size,root=root_path)
        test_data_loader = get_loader_pair(opt.test, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size,root=root_path)
    elif opt.parse is False:
        root_path = './data/origin/train'
        train_data_loader = get_loader(opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, root=root_path)
        val_data_loader = get_loader(opt.val, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, root=root_path)
        test_data_loader = get_loader(opt.test, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, root=root_path)
        if opt.adv is not None:
           adv_data_loader = get_loader_unsupervised(opt.adv, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, root=root_path)
    else:
        root_path = './data/' + opt.parse_path + '/train'
        train_data_loader = get_loader_parse(opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, root=root_path)
        val_data_loader = get_loader_parse(opt.val, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, root=root_path)
        test_data_loader = get_loader_parse(opt.test, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, root=root_path)
        if opt.adv is not None:
           adv_data_loader = get_loader_unsupervised_parse(opt.adv, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, root=root_path)


    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError
    if opt.adv is not None:
        d = Discriminator(opt.hidden_size)
        #d = Discriminator(opt.hidden_size - 2 * opt.entity_size)
        framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader, adv_data_loader, adv=opt.adv, d=d)
    else:
        framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
        
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    prefix = '-'.join([timestamp, model_name, encoder_name, opt.train, opt.val, str(N), str(K)])
    if opt.adv is not None:
        prefix += '-adv_' + opt.adv
    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    
    if model_name == 'proto':
        model = Proto(sentence_encoder, shots =K, hidden_size=opt.hidden_size, drop_rate = opt.dropout)
    elif model_name == 'hatt':
        model = ProtoHATT(sentence_encoder, shots =K, hidden_size=opt.hidden_size, drop_rate = opt.dropout)
    elif model_name == 'multi':
        if encoder_name == 'cnn':
            sentence_encoder = CNNSentenceEncoder(
                    glove_mat,
                    glove_word2id,
                    max_length,
                    mode = 'context',
                    hidden_size=opt.hidden_size - 2*opt.entity_size,
                    mask = True, #mask entity
                    parse =opt.parse,
                    word_embedding_dim=opt.word_emb,
                    pos_embedding_dim=opt.pos_emb
                    )
            entity_encoder = CNNSentenceEncoder(
                    glove_mat,
                    glove_word2id,
                    max_length,
                    mode = 'entity',
                    hidden_size = opt.entity_size,
                    mask = False,
                    parse = opt.parse,
                    word_embedding_dim=opt.word_emb,
                    pos_embedding_dim=opt.pos_emb
                    )
        model = Multi(sentence_encoder, shots = K, hidden_size=opt.hidden_size, entity_encoder = entity_encoder,
                      hybird_attention = opt.hybird_attention, drop_rate = opt.dropout)
    elif model_name == 'pair':
        model = Pair(sentence_encoder, hidden_size = opt.hidden_size)
    else:
        raise NotImplementedError
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    print(f"Checkpoint: {ckpt}")

    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    begin_total = time.time()
    if not opt.only_test:
        if encoder_name == 'bert':
            bert_optim = True
        else:
            bert_optim = False

        framework.train(
                model, prefix,
                batch_size, trainN, N, K, Q,
                train_iter=opt.train_iter,
                val_iter=opt.val_iter,
                val_step=opt.val_step,
                bert_optim=bert_optim,
                pytorch_optim=pytorch_optim,
                learning_rate=opt.lr,
                lr_step_size=opt.lr_step_size,
                weight_decay=opt.weight_decay,
                adv_dis_lr=opt.adv_dis_lr,
                adv_enc_lr=opt.adv_enc_lr,
                warmup_step=opt.warmup_step,
                load_ckpt=opt.load_ckpt,
                save_ckpt=ckpt,
                na_rate=opt.na_rate,
                fp16=opt.fp16,
                pair=opt.pair
                )
    else:
        ckpt = opt.load_ckpt
    end_total = time.time()
    print(f"total training time : {(end_total-begin_total):.2f}s")


    if opt.cluster:
        if opt.parse is False:
            root_path = './data/origin/train'
            unlabel_data_loader = get_loader_all_unsupervised(opt.adv, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, root=root_path)
            origin_train_path = os.path.join(root_path, opt.train + ".json")
            origin_unsupervised_path = os.path.join(root_path, opt.adv + ".json")
        else:
            root_path = './data/'+opt.parse_path+'/train'
            unlabel_data_loader = get_loader_all_unsupervised_parse(opt.adv, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, root=root_path)
            origin_train_path =os.path.join(root_path, opt.train + "_parse.json")
            origin_unsupervised_path = os.path.join(root_path, opt.adv + "_parse.json")
        framework.cluster(model, ckpt, unlabel_data_loader, opt.n_clusters, opt.pseudo_pth, opt.feature_pth, opt.M,origin_unsupervised_path,origin_train_path,root_path)
        print()
        print("cluster over")
        print()


    # Test on official manualscript
    if HP.test_online is not None:
        for n in [5,10]:
            for k in [1,5]:
                print(f"{n}-way-{k}-shot")
                test_file = HP.test_online+'-'+str(n)+'-' +str(k)
                if opt.pair:
                    root_path = './data/origin/test'
                    test_online_loader = get_loader_test_pair(test_file, sentence_encoder, N=n, K=k, Q = 1, na_rate=opt.na_rate,
                                              batch_size=100, root=root_path)
                elif opt.parse is True:
                    root_path = './data/'+opt.parse_path+'/test'
                    test_online_loader = get_loader_test_parse(test_file, sentence_encoder, N=n, K=k, Q = 1, na_rate=opt.na_rate,
                                              batch_size=100, root=root_path)
                else:
                    root_path = './data/origin/test'
                    test_online_loader = get_loader_test(test_file, sentence_encoder, N=n, K=k, Q = 1, na_rate=opt.na_rate,
                                              batch_size=100, root=root_path)
                framework.test_data_loader = test_online_loader
                # test for each configuration
                #result = framework.eval(model, 100, HP.test_N, HP.test_K, 1, 100, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair,test_online = True)

                #test for four configurations
                begin_total = time.time()
                result = framework.eval(model, 100, n, k, 1, 100, na_rate=opt.na_rate, ckpt=ckpt,pair=opt.pair, test_online=True)
                end_total = time.time()
                print(f"{n}-way-{k}-shot test time : {(end_total-begin_total):.2f}s")
                pred_path = os.path.join(root_path,"pred-"+str(n)+"-"+str(k)+".json")
                f = open(pred_path,'w',encoding = 'utf-8')
                f.writelines(json.dumps(result))

    # Test on local file
    else:
        for n in [5,10]:
            #for k in range(1,11):
            if n in [5,10]:
                k_range = [5,10]
            else:
                k_range = [i for i in range(1,11)]
            for k in k_range:
                if opt.parse is True:
                    root_path = './data/'+opt.parse_path+'/train'
                    test_data_loader = get_loader_parse(opt.test, sentence_encoder, N=n, K=k, Q=Q, na_rate=opt.na_rate,
                                              batch_size=batch_size, root=root_path)
                else:
                    root_path = './data/origin/train'
                    test_data_loader = get_loader(opt.test, sentence_encoder, N=n, K=k, Q=Q, na_rate=opt.na_rate,
                                              batch_size=batch_size, root=root_path)
                framework.test_data_loader = test_data_loader
                acc = framework.eval(model, batch_size, n, k, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt,
                                     pair=opt.pair, test_online = False)
                print(f"{n}-way-{k}-shot accuracy : {acc * 100:.2f}")


if __name__ == "__main__":
    for i in range(0,1):
        main()
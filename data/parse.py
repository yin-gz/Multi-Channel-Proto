from supar import Parser
from datetime import datetime
#import nltk
import numpy as np
import os
import json
import time

parser_dep = Parser.load('biaffine-dep-en')

def split_list(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]

#find x pos range in y
def x_in_y(query, base):
    try:
        l = len(query)
    except TypeError:
        l = 1
        query = type(base)((query,))

    for i in range(len(base)):
        if base[i:i+l] == query or base[i:i+l] in query or query in base[i:i+l]:
                return [k for k in range(i,i+l)]
    print('wrong！')
    return [0]


#filter our stop words
def token_filter(tokens, subj_pos, obj_pos):
    subj_word_list = tokens[subj_pos[0]:subj_pos[-1] + 1]
    obj_word_list = tokens[obj_pos[0]:obj_pos[-1] + 1]

    # token filter
    token_new = [token for index, token in enumerate(tokens) if
                 ((token not in ['a', 'an', 'the', 'The', '.', '\\', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/',
                                 '0']) and ('\n' not in token)) or \
                 (index in [i for i in range(subj_pos[0], subj_pos[-1] + 1)]) or \
                 (index in [i for i in range(obj_pos[0], obj_pos[-1] + 1)])]

    subj_pos_new = x_in_y(subj_word_list, token_new)
    obj_pos_new = x_in_y(obj_word_list, token_new)
    return token_new, subj_pos_new, obj_pos_new


#get path to root(one node)
def get_path_root(begin_node, arcs):
    result_path = [begin_node]
    next = arcs[begin_node]
    while(next != 0):
        result_path.append(next-1)
        next = arcs[next-1]
    result_path.append(-1)
    return result_path

#get paths of all nodes to root, retur a list of paths
def get_paths_root(begin_nodes, arcs):
    result_paths = []
    for node in begin_nodes:
        flag = 0
        path = [node]
        next = arcs[node]
        while (next != 0):
            if (next-1) in begin_nodes:
                flag = 1
                break
            else:
                path.append(next-1)
                next = arcs[next - 1]
        if flag == 1:
            continue
        else:
            path.append(-1)
            result_paths.append(path)
    return result_paths

#calculate the distance to common ancestor
def get_ca_length(aim_father_path,father_path_node):
    dis_source = 0
    cas = -1
    for node in father_path_node:
        if (node in aim_father_path) and (node!= -1):
            cas = node
            break
        else:
            dis_source += 1
    dis_aim = aim_father_path.index(cas)
    return (dis_source+dis_aim)

#calculate the shortest distance to subj/obj based on subj paths and arcs
def get_short_dis(aim_father_all, arcs, token_pos, aim_pos):
    dis_to_subj = 1E100
    #check if node_i in aim_node's father path
    for aim_father_path in aim_father_all:
        try:
            path_index = aim_father_path.index(token_pos)
        except:
            continue
        dis_to_subj = round(float(min(dis_to_subj, path_index)),1)
    #True: return
    if dis_to_subj != 1E100:
        return dis_to_subj
    #False:　get the father list of the node and find the common ancestor between the node and subj/obj
    else:
        father_path_node = get_path_root(token_pos, arcs)
        #if aim_node in list
        for index, father_node in enumerate(father_path_node):
            if father_node in aim_pos:
                return index
        #else: calculate the distance to common ancestor
        for aim_father_path in aim_father_all:
            dis_to_subj = round(float(min(dis_to_subj, get_ca_length(aim_father_path,father_path_node))),1)
        return dis_to_subj

def generate_short_dis(arc,subj_pos,obj_pos):
    # get father of subj words
    subj_father_paths = get_paths_root(subj_pos, arc)
    # get father of obj words
    obj_father_paths = get_paths_root(obj_pos, arc)
    subj_dis = []
    obj_dis = []
    #calculate the shortest path between node_i and subj/obj
    for token_pos in range(len(arc)):
        #distance to subj
        if token_pos in subj_pos:
            dis_to_subj = 0.0
        else:
            dis_to_subj = round(float(get_short_dis(subj_father_paths, arc, token_pos, subj_pos)),1)
        subj_dis.append(dis_to_subj)
        #distance to obj
        if token_pos in obj_pos:
            dis_to_obj = 0.0
        else:
            dis_to_obj = get_short_dis(obj_father_paths, arc, token_pos, obj_pos)
        obj_dis.append(dis_to_obj)
    return subj_dis,obj_dis

def generate_deprel(arc,rel,subj_pos,obj_pos):
    subj_deprel = ['None' for i in range(len(arc))]  # deprel type to subj
    obj_deprel = ['None' for i in range(len(arc))]  # deprel type to obj

    for i in subj_pos:
        if arc[i] != 0:
            subj_deprel[arc[i]-1] = rel[i] if rel[i]!='nsubjpass' else 'nsubj'
    for i in obj_pos:
        if arc[i] != 0:
            obj_deprel[arc[i]-1] = rel[i] if rel[i]!='nsubjpass' else 'nsubj'

    for i in range(len(arc)):
        if (arc[i]-1) in subj_pos:
            subj_deprel[i] = rel[i]
        if (arc[i]-1) in obj_pos:
            obj_deprel[i] = rel[i]

    return subj_deprel,obj_deprel

def parse_list(tokens_list,subj_pos_list,obj_pos_list,pos = False):
    result_list = []
    tokens_list_filter = []
    subj_pos_list_filter = []
    obj_pos_list_filter = []
    for (token, subj_pos, obj_pos) in zip(tokens_list, subj_pos_list, obj_pos_list):
        token, subj_pos, obj_pos = token_filter(token, subj_pos, obj_pos)
        tokens_list_filter.append(token)
        subj_pos_list_filter.append(subj_pos)
        obj_pos_list_filter.append(obj_pos)

    #dep_parse
    dep_results = parser_dep.predict(tokens_list_filter, verbose=False)
    arcs = dep_results.arcs
    rels = dep_results.rels
    for (tokens,arc,rel,subj_pos,obj_pos) in zip(tokens_list_filter,arcs,rels,subj_pos_list_filter,obj_pos_list_filter):
        result = {'tokens': [], 'h': [], 't': [], 'subj_deprel': [], 'obj_deprel': [], 'subj_dis': [], 'obj_dis': [],"pos":[]}
        subj_dis, obj_dis = generate_short_dis(arc, subj_pos, obj_pos)
        result['subj_dis'] = subj_dis
        result['obj_dis'] = obj_dis
        subj_deprel, obj_deprel = generate_deprel(arc,rel,subj_pos,obj_pos)
        result['subj_deprel'] = subj_deprel
        result['obj_deprel'] = obj_deprel
        result['tokens'] = tokens
        subj_word_list = tokens[subj_pos[0]:subj_pos[-1] + 1]
        obj_word_list = tokens[obj_pos[0]:obj_pos[-1] + 1]
        result['h'] = [' '.join(subj_word_list),'head',[x_in_y(subj_word_list, tokens)]]
        result['t'] = [' '.join(obj_word_list),'tail',[x_in_y(obj_word_list, tokens)]]
        #if pos == True:
            #pos = [v for (word,v) in nltk.pos_tag(tokens)]
        #else:
            #pos = None
        result_list.append(result)

    return result_list


if __name__ == "__main__":
    parse_path = "./origin_parse"
    suffix = '_parse.json'

    #parse train and val

    path_train = "./origin/train"
    files = os.listdir(path_train)
    for file in files:
        file_name = file.split('.')[0]
        print(file)
        f_out = open(parse_path + "/train/" + file_name + suffix,'w',encoding='utf-8')

        if file.split('_')[0] == 'train' or file.split('_')[0] == 'val':
            f = open(path_train + "/" + file,'r',encoding='utf-8')
            data = json.load(f)
            result = {}
            for rel_type in data:
                train_list = data[rel_type]
                tokens_all = []
                subj_pos_all = []
                obj_pos_all = []
                for j in train_list:
                    tokens_all.append(j['tokens'])
                    subj_pos_all.append(j['h'][-1][0])
                    obj_pos_all.append(j['t'][-1][0])
                parse_rel_type = parse_list(tokens_all, subj_pos_all, obj_pos_all)
                result[rel_type] = parse_rel_type
            f_out.write(json.dumps(result))

        elif file.split('_')[0] == 'unsupervised':
            f = open(path_train + "/" + file,'r',encoding='utf-8')
            data = json.load(f)
            result = []
            tokens_all = []
            subj_pos_all = []
            obj_pos_all = []
            for instance in data:
                tokens_all.append(instance['tokens'])
                subj_pos_all.append(instance['h'][-1][0])
                obj_pos_all.append(instance['t'][-1][0])
            result = parse_list(tokens_all, subj_pos_all, obj_pos_all)
            f_out.write(json.dumps(result))



    #parse test
    UNION_NUM = 1000
    path_test = "./origin/test"
    files = os.listdir(path_test)
    for file in files:
        file_name = file.split('.')[0]
        K = int(file_name.split('-')[-1])
        N = int(file_name.split('-')[-2])
        if K!=5:
            continue
        print(file)
        total_time = 0
        f_out = open(parse_path + "/test/" + file_name + suffix,'w',encoding='utf-8')
        f = open(path_test + "/" + file,'r',encoding='utf-8')
        data = json.load(f)
        count = 0
        tokens_mtrain_all = []
        subj_pos_mtrain_all = []
        obj_pos_mtrain_all = []
        tokens_mtest_all = []
        subj_pos_mtest_all = []
        obj_pos_mtest_all = []
        times = 0

        for instance in data:
            count += 1
            train_list = instance['meta_train']
            test = instance['meta_test']
            #get UNION_NUM*N support instances
            for K_list in train_list:
                for j in K_list:
                    tokens_mtrain_all.append(j['tokens'])
                    subj_pos_mtrain_all.append(j['h'][-1][0])
                    obj_pos_mtrain_all.append(j['t'][-1][0])
            tokens_mtest_all.append(test['tokens'])
            subj_pos_mtest_all.append(test['h'][-1][0])
            obj_pos_mtest_all.append(test['t'][-1][0])

            # begin to parse
            if count == UNION_NUM:
                times += 1
                print('Already Parse:',count, 'Left:', len(data) - times*count)

                parse_mtrain_all = parse_list(tokens_mtrain_all, subj_pos_mtrain_all, obj_pos_mtrain_all)
                parse_mtest_all = parse_list(tokens_mtest_all, subj_pos_mtest_all, obj_pos_mtest_all)
                parse_mtrain_split = split_list(parse_mtrain_all, int(len(parse_mtrain_all)/UNION_NUM))

                #return to N way K shot
                i = 0
                for parse_mtrain in parse_mtrain_split:
                    parse_mtest = parse_mtest_all[i]
                    parse_mtrain_result = []
                    parse_mtrain_split = split_list(parse_mtrain, int(len(parse_mtrain)/N))
                    for  parse_mtrain_K in parse_mtrain_split:
                        parse_mtrain_result.append(parse_mtrain_K)
                    result_line = {'meta_test':parse_mtest,'meta_train':parse_mtrain_result}
                    i += 1
                    f_out.writelines(json.dumps(result_line) + '\n')
                count = 0
                tokens_mtrain_all = []
                subj_pos_mtrain_all = []
                obj_pos_mtrain_all = []
                tokens_mtest_all = []
                subj_pos_mtest_all = []
                obj_pos_mtest_all = []



import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import time


class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        print(path)
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):

        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                       item['h'],
                                                       item['t'])
        return word, pos1, pos2, mask

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        if len(self.classes) > 64:
            # use only when retraining
            target_classes1 = random.sample(self.classes[:-10], self.N - 1)
            target_classes2 = random.sample(self.classes[-10:], 1)
            target_classes = target_classes1 + target_classes2
        else:
            target_classes = random.sample(self.classes, self.N)

        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}

        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,
                                 self.classes))

        #First, sample self.N*self.Q query samples (every labeling with 0-self.N-1)
        query_label_all = random.choices([i for i in range(self.N)], k = self.N*self.Q)
        #Then, count query samples of each labeling
        query_label_count = {k: query_label_all.count(k) for k in range(self.N)}

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))),
                    self.K + query_label_count[i], False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(
                        self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask)
                count += 1

            query_label += [i] * query_label_count[i]

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                list(range(len(self.json_data[cur_class]))),
                1, False)[0]
            word, pos1, pos2, mask = self.__getraw__(
                self.json_data[cur_class][index])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(query_set, word, pos1, pos2, mask)
        query_label += [self.N] * Q_na

        #Shuffle
        query_labels_new = []
        batch_query_new = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        rand_index = [i for i in range(len(query_label))]
        random.shuffle(rand_index)
        for index in rand_index:
            query_labels_new.append(query_label[index])
            for key in query_set:
                batch_query_new[key].append(query_set[key][index])
        query_set = batch_query_new
        query_label = query_labels_new

        return support_set, query_set, query_label

    def __len__(self):
        return 1000000000


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)

    return batch_support, batch_query, batch_label


def get_loader(name, encoder, N, K, Q, batch_size,
               num_workers=0, collate_fn=collate_fn, na_rate=0, root='./data'):
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)


class FewRelDatasetPair(data.Dataset):
    """
    FewRel Pair Dataset
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.max_length = encoder.max_length

    def __getraw__(self, item):
        word = self.encoder.tokenize(item['tokens'],
                                     item['h'],
                                     item['t'])
        return word

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support = []
        query = []
        fusion_set = {'word': [], 'mask': [], 'seg': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,
                                 self.classes))

        query_label_all = random.choices([i for i in range(self.N)], k=self.N * self.Q)
        query_label_count = {k: query_label_all.count(k) for k in range(self.N)}

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                list(range(len(self.json_data[class_name]))),
                self.K + query_label_count[i], False)
            count = 0
            for j in indices:
                word = self.__getraw__(
                    self.json_data[class_name][j])
                if count < self.K:
                    support.append(word)
                else:
                    query.append(word)
                count += 1

            query_label += [i] * query_label_count[i]

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                list(range(len(self.json_data[cur_class]))),
                1, False)[0]
            word = self.__getraw__(
                self.json_data[cur_class][index])
            query.append(word)
        query_label += [self.N] * Q_na

        # shuffle query set
        query_labels_new = []
        query_new = []
        rand_index = [i for i in range(len(query_label))]
        # seed = str(time.time())[-4:]
        # random.seed(seed)
        random.shuffle(rand_index)
        for index in rand_index:
            query_labels_new.append(query_label[index])
            query_new.append(query[index])
        query = query_new
        query_label = query_labels_new

        for word_query in query:
            for word_support in support:
                if self.encoder_name == 'bert':
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])
                    word_tensor = torch.zeros((self.max_length)).long()
                else:
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['</s>'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['<s>'])
                    word_tensor = torch.ones((self.max_length)).long()
                new_word = CLS + word_support + SEP + word_query + SEP
                for i in range(min(self.max_length, len(new_word))):
                    word_tensor[i] = new_word[i]
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                seg_tensor = torch.ones((self.max_length)).long()
                seg_tensor[:min(self.max_length, len(word_support) + 1)] = 0
                fusion_set['word'].append(word_tensor)
                fusion_set['mask'].append(mask_tensor)
                fusion_set['seg'].append(seg_tensor)

        return fusion_set, query_label

    def __len__(self):
        return 1000000000


def collate_fn_pair(data):
    batch_set = {'word': [], 'seg': [], 'mask': []}
    batch_label = []
    fusion_sets, query_labels = zip(*data)
    for i in range(len(fusion_sets)):
        for k in fusion_sets[i]:
            batch_set[k] += fusion_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_set:
        batch_set[k] = torch.stack(batch_set[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_set, batch_label


def get_loader_pair(name, encoder, N, K, Q, batch_size,
        num_workers=0, collate_fn=collate_fn_pair, na_rate=0, root='./data', encoder_name='bert'):
    dataset = FewRelDatasetPair(name, encoder, N, K, Q, na_rate, root, encoder_name)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)


class FewRelUnsupervisedDataset(data.Dataset):
    """
    FewRel Unsupervised Dataset
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        print(path)
        self.json_data = json.load(open(path))
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                       item['h'],
                                                       item['t'])
        return word, pos1, pos2, mask

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        total = self.N * self.K
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}

        indices = np.random.choice(list(range(len(self.json_data))), total, False)
        for j in indices:
            word, pos1, pos2, mask = self.__getraw__(
                self.json_data[j])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(support_set, word, pos1, pos2, mask)

        return support_set

    def __len__(self):
        return 1000000000


def collate_fn_unsupervised(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    support_sets = data
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    return batch_support


def get_loader_unsupervised(name, encoder, N, K, Q, batch_size,
                            num_workers=0, collate_fn=collate_fn_unsupervised, na_rate=0, root='./data'):
    dataset = FewRelUnsupervisedDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)


class FewRelAllUnsupervisedDataset(data.Dataset):
    """
    FewRel All Unsupervised Dataset
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                       item['h'],
                                                       item['t'])
        return word, pos1, pos2, mask

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}

        indices = list(range(len(self.json_data)))
        for j in indices:
            word, pos1, pos2, mask = self.__getraw__(
                self.json_data[j])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(support_set, word, pos1, pos2, mask)

        return support_set

    def __len__(self):
        return 1000000000


def collate_fn_all_unsupervised(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    support_sets = data
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    return batch_support


def get_loader_all_unsupervised(name, encoder, N, K, Q, batch_size=1,
                                num_workers=0, collate_fn=collate_fn_all_unsupervised, na_rate=0, root='./data'):
    dataset = FewRelAllUnsupervisedDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)


class FewRelSampleUnsupervisedDataset(data.Dataset):
    """
    FewRel Unsupervised Dataset
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                       item['h'],
                                                       item['t'])
        return word, pos1, pos2, mask

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        total = len(self.json_data)
        index = np.random.randint(0, total, (1))[0]
        word, pos1, pos2, mask = self.__getraw__(
            self.json_data[index])
        word = torch.tensor(word).long()
        pos1 = torch.tensor(pos1).long()
        pos2 = torch.tensor(pos2).long()
        mask = torch.tensor(mask).long()
        self.__additem__(support_set, word, pos1, pos2, mask)

        return support_set

    def __len__(self):
        return 1000000000


def collate_fn_sample_unsupervised(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    support_sets = data
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)  # batch_support[k]长度为40
    return batch_support


def get_loader_sample_unsupervised(name, encoder, N, K, Q, batch_size,
                                   num_workers=0, collate_fn=collate_fn_sample_unsupervised, na_rate=0, root='./data'):
    dataset = FewRelSampleUnsupervisedDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=256,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn_sample_unsupervised)
    return iter(data_loader)


class FewRelTestDataset(data.Dataset):
    """
    FewRel All Unsupervised Dataset
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                       item['h'],
                                                       item['t'])
        return word, pos1, pos2, mask

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}

        meta_train = self.json_data[index]['meta_train']
        meta_test = self.json_data[index]['meta_test']
        for N_way in meta_train:
            for train_example in N_way:
                word, pos1, pos2, mask = self.__getraw__(
                    train_example)
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                self.__additem__(support_set, word, pos1, pos2, mask)
        word, pos1, pos2, mask = self.__getraw__(
            meta_test)
        word = torch.tensor(word).long()
        pos1 = torch.tensor(pos1).long()
        pos2 = torch.tensor(pos2).long()
        mask = torch.tensor(mask).long()
        self.__additem__(query_set, word, pos1, pos2, mask)

        return support_set, query_set

    def __len__(self):
        return len(self.json_data)


def collate_fn_test(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    support_sets, query_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    return batch_support, batch_query


def get_loader_test(name, encoder, N, K, Q, batch_size=1,
                    num_workers=0, collate_fn=collate_fn_test, na_rate=0, root='./data'):
    dataset = FewRelTestDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)


class FewRelDatasetTestPair(data.Dataset):
    """
    FewRel Pair Dataset
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.max_length = encoder.max_length

    def __getraw__(self, item):
        word = self.encoder.tokenize(item['tokens'],
                                     item['h'],
                                     item['t'])
        return word

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        support = []
        query = []
        fusion_set = {'word': [], 'mask': [], 'seg': []}
        meta_train = self.json_data[index]['meta_train']
        meta_test = self.json_data[index]['meta_test']
        for N_way in meta_train:
            for train_example in N_way:
                word = self.__getraw__(train_example)
                support.append(word)
        word = self.__getraw__(
            meta_test)
        query.append(word)

        for word_query in query:
            for word_support in support:
                SEP = self.encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])
                CLS = self.encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])
                new_word = CLS + word_support + SEP + word_query + SEP
                word_tensor = torch.zeros((self.max_length)).long()
                for i in range(min(self.max_length, len(new_word))):
                    word_tensor[i] = new_word[i]
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                seg_tensor = torch.ones((self.max_length)).long()
                seg_tensor[:min(self.max_length, len(word_support) + 1)] = 0
                fusion_set['word'].append(word_tensor)
                fusion_set['mask'].append(mask_tensor)
                fusion_set['seg'].append(seg_tensor)

        return fusion_set

    def __len__(self):
        return len(self.json_data)

def collate_fn_test_pair(data):
    batch_set = {'word': [], 'seg': [], 'mask': []}
    fusion_sets = data
    for i in range(len(fusion_sets)):
        for k in fusion_sets[i]:
            batch_set[k] += fusion_sets[i][k]
    for k in batch_set:
        batch_set[k] = torch.stack(batch_set[k], 0)
    return batch_set


def get_loader_test_pair(name, encoder, N, K, Q, batch_size,
                    num_workers=0, collate_fn = collate_fn_test_pair, na_rate=0, root='./data'):
    dataset = FewRelDatasetTestPair(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)
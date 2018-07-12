import torchvision
from torchvision import transforms
import torch
import os
import numpy as np
from preprocess import mel_spec
import random
random.seed(233)


data_dir = "/home/zfhu/playground/DSP/data/"
class_num = 20
item_num = 20


class DataFeed(object):
    def __init__(self):
        self.data_dir = data_dir
        self.stuids = os.listdir(self.data_dir)
        self.cates = "语音 余音 识别 失败 中国 忠告 北京 背景 上海 商行 复旦 饭店 Speech Speaker Signal File Print Open Close Project".split(
            ' ')

    def __len__(self):
        return len(self.stuids)

    def get_path(self, stu, cate, ith):
        assert 0 <= stu < len(self)
        assert 0 <= cate < 20
        assert 0 <= ith < 20
        stuid = self.stuids[stu]
        ret = os.path.join(self.data_dir, stuid, "{0}-{1:02}-{2:02}.wav".format(stuid, cate, ith + 2))
        return ret

    def get_blob(self, stu, cate, ith):
        path = self.get_path(stu, cate, ith)
        return open(path, "rb").read()

    def get_stu(self, stu):
        paths = []
        for i in range(class_num):
            for j in range(item_num):
                paths.append((self.get_path(stu, i, j), i))
        return paths, self.stuids[stu]

    def get_by_id(self, num):
        ith = num % 20
        num //= 20
        cate = num % 20
        num //= 20
        assert num < len(self)
        return self.get_path(num, cate, ith), cate


def read_sample(path):
    try:
        spec = mel_spec(path)
        spec = spec.reshape(1, spec.shape[0], -1)
        spec = torch.from_numpy(spec).type(torch.float)
    except:
        raise IOError("你他娘的是有毒还是咋滴？ {}".format(path))

    return spec


class SpecDataset(torch.utils.data.Dataset):

    def __init__(self, paths, cuda=True):
        self.paths = []
        self.cuda = cuda
        for t in paths:
            self.paths += t

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        # print(item)
        # print(self.paths[item])
        (path, clas) = self.paths[item]
        spec = read_sample(path)
        return spec, torch.tensor(clas)


def spec_folder(candidates, nfolds=10):
    d = DataFeed()
    paths = []
    for c in candidates:
        for i in range(class_num):
            for j in range(item_num):
                paths.append((d.get_path(c, i, j), i))
    random.shuffle(paths)
    tot = len(paths)
    fold_size = tot // nfolds
    folds = [paths[i: i + fold_size] for i in range(0, tot, fold_size)]
    return folds


def spec_cvloader(folds, nfold, batch_size, num_workers=32, shuffle=True, cuda=True):
    return torch.utils.data.DataLoader(SpecDataset(folds[:nfold] + folds[nfold + 1:]), pin_memory=cuda,
                                      batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), \
           torch.utils.data.DataLoader(SpecDataset(folds[nfold: nfold + 1]), batch_size=batch_size, pin_memory=cuda,
                                       shuffle=shuffle, num_workers=num_workers)


def spec_loader(candidates, batch_size, num_workers=32, shuffle=True, cuda=True):
    d = DataFeed()
    paths = []
    for c in candidates:
        for i in range(class_num):
            for j in range(item_num):
                paths.append((d.get_path(c, i, j), i))
    return torch.utils.data.DataLoader(SpecDataset([paths]), pin_memory=cuda,
                                      batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)





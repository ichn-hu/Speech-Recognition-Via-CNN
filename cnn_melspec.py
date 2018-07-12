import torch
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
import argparse

from model.conv import vgg11_bn
from preprocess.dataset import *


xwriter = SummaryWriter('cnn_melspec_log')
data_feed = DataFeed()


def train(model: torch.nn.Module, optimizer, spec_fd, nepoch, nbatch=32):
    criterion = nn.CrossEntropyLoss().cuda()
    losses = []
    model.train()
    print("start train")

    for iepoch in range(nepoch):
        train_iter, val_iter = spec_cvloader(spec_fd, iepoch % len(spec_fd), nbatch)
        # acc = evaluate(model, val_iter)
        for i, (X, Y) in enumerate(train_iter):
            # print(X.shape, Y.shape)
            X = X.cuda()
            Y = Y.cuda()
            Ym = model(X)
            loss = criterion(Ym, Y)
            xwriter.add_scalar('train/{}th'.format(iepoch), loss.item() / X.size(0), i)
            losses.append(loss.item() / X.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate(model, val_iter)
        print("Loss: {:.3f} Acc: {:.3f}".format(losses[-1], acc))

    print("train finished")


def evaluate(model: torch.nn.Module, val_iter):
    model.eval()
    acc, tot = 0, 0
    with torch.no_grad():
        for i, (X, Y) in enumerate(val_iter):
            X = X.cuda()
            Y = Y.cuda()
            Ym = model(X)
            Ym = torch.argmax(Ym, dim=1).view(-1)
            Y = Y.view(-1)
            tot += Ym.size(0)
            acc += (Ym == Y).sum().item()
    return acc / tot


def individual_test(model: torch.nn.Module, stu):
    iter = spec_loader([stu], 32)
    acc = evaluate(model, iter)
    print("outsider test acc: {:.3f}".format(acc))


def outsider_test(model: torch.nn.Module, outsiders):
    for o in outsiders:
        iter = spec_loader([o], 32)
        acc = evaluate(model, iter)
        print("outsider {} test acc: {:.3f}".format(data_feed.stuids[o], acc))


def infer(model: torch.nn.Module, sample_path):
    X = read_sample(sample_path)
    X = X[None, :, :, :]
    X = X.cuda()
    model.eval()
    print(X)
    print(X.shape)
    Ym = model(X)
    print(Ym)
    return data_feed.cates[torch.argmax(Ym, dim=1).item()]


def build_model(load=''):
    model = vgg11_bn()
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)
    if load:
        checkpoint = torch.load(load)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    model.cuda()
    return model, optimizer


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--name", default="cnn_melspec", type=str)
    argparser.add_argument("--infer", default='', type=str)
    argparser.add_argument("--nepoch", default=10, type=int)
    argparser.add_argument("--save", default="save.ptr", type=str)
    argparser.add_argument("--load", default='', type=str)
    args = argparser.parse_args()


    model, optimizer = build_model(args.load)
    if args.infer:
        infer(model, args.infer)

    candidates = range(22)
    outsiders = range(32)

    spec_fd = spec_folder(candidates, 10)

    train(model, optimizer, spec_fd, args.nepoch)
    xwriter.export_scalars_to_json("./test.json")
    xwriter.close()

    # outsider_test(model, outsiders)

    checkpointer = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(checkpointer, args.save)


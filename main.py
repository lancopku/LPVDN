import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from evaluate import cluster, evaluate
import numpy as np
import json
import math
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import _joint_probabilities
from scipy.spatial.distance import squareform
from tqdm import tqdm
import argparse
from draw import draw
from model import LPVDN

def get_parser():
    parser = argparse.ArgumentParser(description='LPVDN Training')
    parser.add_argument('--batch_size', default=800, type=int)
    parser.add_argument('--datadir', default='./data/', type=str)
    parser.add_argument('--nClusters', default=10, type=int)
    parser.add_argument('--hid_dim', default=10, type=int)
    parser.add_argument('--n_epoch', default=300, type=int)
    parser.add_argument('--save_iter', default=10, type=int)
    parser.add_argument('--alpha0', default=1, type=float)
    parser.add_argument('--alpha1', default=1e-4, type=float)
    parser.add_argument('--savedir', default='./model/', type=str)
    parser.add_argument('--logdir', default='./log/', type=str)
    return parser

def cal_prob(x):
    x = x.reshape(x.shape[0], -1)
    dists = pairwise_distances(x, squared=True)
    ppl = len(x)*0.01
    p = _joint_probabilities(dists, ppl, 0)
    return squareform(p)

def train(epoch):
    lr_s.step()
    for batch_idx, ((data, _), (neg, _)) in enumerate(zip(tqdm(train_loader, ascii=True), neg_loader)):
        if args.cuda:
            data = data.cuda()
            neg = neg.cuda()
        data = data.view(data.size(0), -1)
        neg = neg.view(neg.size(0), -1)
        ELBO_loss, z, mean, logvar = model.ELBO_Loss(data, bce=True)
        MIloss = model.MIloss(z, data, neg)
        global_loss = ELBO_loss + args.alpha0*MIloss
        
        out = model.t_encoder(mean)
        p = torch.FloatTensor(cal_prob(mean.cpu().data.numpy())).cuda()
        dists = (out.unsqueeze(0)-out.unsqueeze(1)).pow(2).sum(2)
        dists = (dists+1).pow(-1)
        mask = torch.eye(data.size(0)).cuda()
        dists = dists*(1-mask)
        q = dists/dists.sum()
        q = q.clamp_min(1e-20)
        local_loss = (p*torch.log(p.clamp_min(1e-20))).sum()-(p*torch.log(q)).sum()
        

        loss = global_loss + local_loss*args.alpha1

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tMI: {:.6f}\tELBO: {:.6f}\tLC: {:.6f}\tLR: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), MIloss.item(),ELBO_loss.item(),local_loss.item(),lr_s.get_lr()[0]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(draw_pic=False):
    print('testing')
    vecs, tgts = [], []
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data = data.view(data.size(0), -1)
        mean, logvar = model.encoder(data)
        vec = model.t_encoder(mean)
        tgts.append(target.cpu().data.numpy())
        vecs.append(vec.cpu().data.numpy())
    vecs = np.vstack(vecs)
    tgts = np.hstack(tgts)
    lbls = cluster(vecs)
    result = evaluate(tgts, lbls)
    print(result)
    if draw_pic:
        draw(vecs, tgts, 'visualize')
    return result

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    train_dataset = datasets.MNIST(root=args.datadir,
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
    test_dataset = datasets.MNIST(root=args.datadir,
                                  train=False,
                                  transform=transforms.ToTensor())

    all_dataset = train_dataset + test_dataset

    train_loader = torch.utils.data.DataLoader(dataset=all_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    neg_loader = torch.utils.data.DataLoader(dataset=all_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=all_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False)

    model = LPVDN(args)
    if args.cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(),lr=2e-3)
    lr_s = StepLR(optimizer,step_size=10,gamma=0.95)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    with open(os.path.join(args.logdir, 'LPVDN.txt'), 'w') as f:
        pass

    pretrain_dir = './pretrained'
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)
    model.pre_train(train_loader, 50, os.path.join(pretrain_dir, 'pretrain_model.pk'))
    for epoch in range(1, args.n_epoch+1):
        model.train()
        train(epoch)
        if epoch%args.save_iter==0:
            model.eval()
            res = test(draw_pic=False)
            res['epoch'] = epoch
            with open(os.path.join(args.logdir, 'LPVDN.txt'), 'a') as f:
                f.write(json.dumps(res)+'\n')
            torch.save(model.state_dict(), os.path.join(args.savedir, 'snapshot_%d.pt'%(epoch)))


from __future__ import print_function
import argparse
import random
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import utils
from torch_geometric.data import Data, Dataset,DataLoader
from torch_scatter import scatter_mean
import torch_geometric.transforms as GT
import math
import json
from model0 import Net
from dataset0 import ScitsrDataset

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=32, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz\'')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=20, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=100, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=10, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
#print(opt)

if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

#cudnn.benchmark = True

#if torch.cuda.is_available() and not opt.cuda:
#    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


root_path = ''
train_dataset = ScitsrDataset(root_path)
# root_path = ''
# test_dataset = ScitsrDataset(root_path)
root_path = ''
eval_dataset = ScitsrDataset(root_path)
# print("samples:",len(train_dataset),len(test_dataset))
print("samples:",len(train_dataset),len(eval_dataset))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#test_loader = DataLoader(ds_test, batch_size=32)

#vob=open("./data/arti-images/vocab_fapiao.txt",'r')
#opt.alphabet=vob.readline()
#converter = utils.strLabelConverter(opt.alphabet)
#criterion = CTCLoss()  # pytorch 0.4
#criterion = torch.nn.CTCLoss    # pytorch 1.0.0
nclass = 2
input_num = 8
print('num of classes:',nclass)

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


device = torch.device("cpu" )
model = Net(input_num,nclass)#.to(device)
model.cuda()

#for k,v in crnn.state_dict().items():
#    print(k)
model.apply(weights_init)
criterion = torch.nn.NLLLoss() 

if opt.cuda:
    model.cuda()
    criterion = criterion.cuda()

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(model.parameters(), lr=opt.lr)
else:
    optimizer = optim.RMSprop(model.parameters(), lr=opt.lr)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    data_loader = DataLoader(dataset, batch_size=32)
    val_iter = iter(data_loader)
    i = 0
    n_correct = 0
    n_total = 0
    loss_avg = utils.averager()
    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        preds = net(data)
        cost = criterion(preds, data.y.cuda())
        loss_avg.add(cost)
        _, preds = preds.max(1)
        label = data.y.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        n_correct=n_correct+(label==preds).sum()
        n_total=n_total+label.shape[0]
        #print("correct:",n_correct,label.shape[0])
    accuracy = n_correct / float(n_total)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(train_iter, net, criterion, optimizer):
    data = train_iter.next()
    preds = net(data)
    #batch_size = data.y.size()[0]
    cost = criterion(preds, data.y.cuda()) 
    net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

if opt.crnn != '':  
    print('loading pretrained model from %s' % opt.crnn)
    crnn.load_state_dict(torch.load(opt.crnn),strict=False)

# 直接val一下。
val(model, eval_dataset, criterion)

for epoch in range(opt.niter):
    train_iter = iter(train_loader)
    i = 0
    print('epoch',epoch, ' dataset size:', len(train_loader))
    
    while i < len(train_loader):
        for p in model.parameters():
            p.requires_grad = True
        model.train()
  
        cost = trainBatch(train_iter, model, criterion, optimizer)
        loss_avg.add(cost)
        
        i += 1
        #print(loss_avg)
        
        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

    if epoch % opt.valInterval == 0:
        val(model, eval_dataset, criterion)#

        # do checkpointing
    if epoch % opt.saveInterval == 0 :
        torch.save(model.state_dict(), '{0}/net_{1}_{2}.pth'.format(opt.experiment, epoch, i))
        #for k,v in crnn.state_dict().items():
        #     print(k)

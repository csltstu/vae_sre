#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: trainer.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Mon 20 Jan 2020 10:52:14 PM CST
# ************************************************************************/

import os
import torch
from data_loader import *
import torch.optim as optim
from utils import *
import torch.nn.functional as F
from vae import *
import kaldi_io

class trainer(object):
    def __init__(self, args):
        self.args = args

        # init model
        self.model = VAE(z_dim=50)

        # init optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # init work env
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        if args.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
            self.device = torch.device("cuda:" + args.device)
        else:
            self.device = torch.device("cpu")
        print("training device: {}".format(self.device))



    def train(self):
        args = self.args

        # init dataloader
        self.dataset = feats_data_loader(npz_path="./data/feats.npz", dataset_name="vox")
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)
        start_epoch = self.reload_checkpoint()
        self.model.to(self.device)

        # main to train
        start_epoch = self.epoch_idx
        for idx in range(start_epoch, args.epochs): # epochs
            self.epoch_idx = idx
            train_loss = 0
            for batch_idx, (data, label) in enumerate(self.train_loader): # batchs

                data = data.to(self.device)
                self.optimizer.zero_grad()

                recon_data, mu, logvar = self.model(data)

                # loss
                # reconstruct loss: either mse or binary_cross_entropy.
                # mse = torch.nn.MSELoss(reduce=True, size_average=False)
                # recon_loss = mse(recon_data, data)
                recon_loss = F.binary_cross_entropy(recon_data, data, size_average=False)
                KLD = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
                loss = KLD + recon_loss

                loss.backward()

                cur_loss = loss.item()

                train_loss += cur_loss
                self.optimizer.step()

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tKLD: {}\tre_con: {}'.format(
                    self.epoch_idx, batch_idx * len(data), len(self.train_loader.dataset),
                    100.*batch_idx / len(self.train_loader),
                    cur_loss/len(data), KLD/len(data), recon_loss.item()/len(data)))

            print('====> Epoch: {} Average loss: {:.4f}'.format(
                self.epoch_idx, train_loss / len(self.train_loader.dataset)))

            self.save_checkpoint()
    

    # generate z
    def generate_z(self):

        # init model
        self.reload_checkpoint()
        self.model.to(self.device)
        self.model.eval()

        # init data x
        dataset = feats_data_loader(npz_path="./data/feats.npz", dataset_name="vox")
        labels = dataset.label

        test_loader = torch.utils.data.DataLoader(dataset, batch_size=200000, shuffle=False)
        for batch_idx, (data, label) in enumerate(test_loader): # batchs
            data = data.to(self.device)
            self.optimizer.zero_grad()
            mu, logvar = self.model.encode(data)
            mu = mu.cpu().detach().numpy()
            if batch_idx == 0:
                feats = mu
            else:
                feats = np.vstack((feats, mu))
            print(batch_idx, " generating... feats shape: ", np.shape(feats))


        np.savez("test.npz", feats=feats, spkers=labels)
        print("sucessfully saved in {}".format("test.npz"))

        
    def reload_checkpoint(self):
        '''check if checkpoint file exists and reload the checkpoint'''
        args = self.args
        self.epoch_idx = 0
        if not os.path.exists(args.ckpt_dir):
            os.mkdir(args.ckpt_dir);
            print("can not find ckpt dir, and creat {} dir".format(args.ckpt_dir))
            print("start to train fron epoch 0...")
        else:
            files = os.listdir(args.ckpt_dir)
            ckpts = []
            for f in files:
                if(f.endswith(".pt")):
                    ckpts.append(f)
            if(len(ckpts)):
                import re
                for ckpt in ckpts:
                    ckpt_epoch = int(re.findall(r"\d+", ckpt)[0])
                    if ckpt_epoch>self.epoch_idx:
                        self.epoch_idx=ckpt_epoch

                checkpoint_dict = torch.load('{}/ckpt_epoch{}.pt'.format(args.ckpt_dir, self.epoch_idx), map_location=self.device)
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.model.load_state_dict(checkpoint_dict['model'])
                print("sucessfully reload mdl_epoch{}.pt".format(self.epoch_idx))
                self.epoch_idx+=1


    def save_checkpoint(self):
        '''save the checkpoint, including model and optimizer, and model index is epoch'''
        args=self.args
        if not os.path.exists(args.ckpt_dir):
            os.mkdir(args.ckpt_dir);
            print("can not find ckpt dir, and creat {} dir".format(args.ckpt_dir))

        PATH = '{}/ckpt_epoch{}.pt'.format(args.ckpt_dir, self.epoch_idx)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
            }, PATH)


if __name__ == "__main__":
	args = get_args()
	vae_trainer = trainer(args)
	vae_trainer.generate_z()

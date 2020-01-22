#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: args.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Mon 20 Jan 2020 10:44:10 PM CST
# ************************************************************************/

import argparse

def get_args():
	parser = argparse.ArgumentParser(description='PyTorch Flows for sre')

	parser.add_argument(
		'--batch_size',
		type=int,
		default=10000,
		help='input batch size for training (default: 10000)')

	parser.add_argument(
		'--epochs',
		type=int,
		default=100,
		help='number of epochs to train (default: 100)')

	parser.add_argument(
		'--lr', type=float, default=0.001, help='learning rate (default: 0.001)')

	parser.add_argument(
		'--dataset',
		default='VOXCELEB1',
		help='VOXCELEB1 | SITW_DEV_ENROLL | SITW_DEV_TEST | SITW_EVAL_ENROLL | SITW_EVAL_TEST')

	parser.add_argument(
		'--flow', default='maf', help='flow to use: maf | realnvp | glow')

	parser.add_argument(
		'--no_cuda',
		action='store_true',
		default=False,
		help='disables CUDA training')

	parser.add_argument(
		'--num_blocks',
		type=int,
		default=10,
		help='number of invertible blocks (default: 10)')

	parser.add_argument(
		'--seed', type=int, default=1, help='random seed (default: 1)')

	parser.add_argument(
		'--device',
		default='0',
		help='cuda visible devices (default: 0)')

	parser.add_argument(
		'--log',
		default='log',
		help='log status')

	parser.add_argument(
		'--interval',
		type=int,
		default=10,
		help='how many epochs to wait before saving models')

	parser.add_argument(
		'--ckpt_dir',
		default='ckpt',
		help='dir to save check points')

	parser.add_argument(
		'--kaldi_dir',
		default='kaldi',
		help='dir to save feats in Kaldi format')

	args = parser.parse_args()
	
	return args


#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: ark2npz.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Sun 19 Jan 2020 11:08:41 AM CST
# ************************************************************************/


import numpy as np
import kaldi_io


def build_utt2spk_dic(utt2spk_path):
	'''build a hash map (utt->spk)'''
	print("start to build dic")
	utt2spk_dic={}
	utt2spk = np.loadtxt(utt2spk_path, dtype=str , comments='#', delimiter=" ")
	for it in utt2spk:
		utt=it[0]
		spk=it[1]
		utt2spk_dic[utt]=spk

	print("building dic is down")
	return utt2spk_dic


def ark2npz(ark_path, npz_path, utt2spk):
	'''load ark data format and save as npz data format

	label: spker.shape=(utt_num, )
	data : feats.shape=(utt_num, 72)

	//load the data and label
	feats = np.load(npz_path, allow_pickle=True)['feats']
	spkers = np.load(npz_path, allow_pickle=True)['spkers']

	'''
	print("ark data loading...")
	utts = []
	mats = []
	for k, v in kaldi_io.read_mat_ark(ark_path):
		utts.append(k)
		mats.append(v)
	utt2spk = build_utt2spk_dic(utt2spk)
	counter = 0
	feats = []
	spkers = []
	for mat in mats:
		for i in mat:
			feats.append(i)
			spkers.append(utt2spk[utts[counter]])
		counter+=1
	
	# convert string-label to num label
	string_lable = spkers
	spkers = np.unique(spkers)

	index = 0
	table = {}
	for it in spkers:
		table[it] = index
		index+=1
	
	num_label=[]
	for spk in string_lable:
		num_label.append(table[spk])
		
	print("saving...")
	np.savez(npz_path, feats=feats, spkers=num_label)
	print("sucessfully convert {} to {} ".format(ark_path, npz_path))
	print("ark->npz down")


if __name__=="__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--src_file', default="feats.ark", help='src file of feats.(ark)')
	parser.add_argument('--dest_file',default="feats.npz", help='dest file of feats.(npz)')
	parser.add_argument('--utt2spk',default="utt2spk", help='utt2spk is used to build utt2spk_dic, which transfers the label from utt to spker')
	args = parser.parse_args()

	ark2npz(args.src_file, args.dest_file, args.utt2spk)

	# test
	print("\n\ntest...\n")
	feats = np.load(args.dest_file, allow_pickle=True)['feats']
	spkers = np.load(args.dest_file, allow_pickle=True)['spkers']

	print("feats shape: ", np.shape(feats))
	print("spker label shape: ", np.shape(spkers))
	print("num of spker: ", np.shape(np.unique(spkers)))

	print(spkers[0])
	print(feats[0])

# 	spk = np.unique(spkers)
# 	for it in spk:
# 		print(it)

import os
import numpy as np
import pandas as pd
import random
import pickle

def load_data(data_path):
	with open(data_path) as f:
		lines=f.readlines()
	print(len(lines))
	sentence_pairs=[]
	label_list=[]
	for line in lines:
		line_split=line.strip().split("\t")
		sentence1=[number for number in line_split[0].strip().split()]
		sentence2=[number for number in line_split[1].strip().split()]
		sentence_pairs.append((sentence1,sentence2))
		if len(line_split)==3:
			label_list.append(int(line_split[-1]))
	if len(line_split)==3:
		return sentence_pairs,label_list
	else:
		return sentence_pairs

def get_word2id(sentence_pairs):
    word2id={}
    word2id['PAD']=len(word2id)
    word2id['UNK']=len(word2id)
    import collections
    all_words=[]
    for sen_pairs in sentence_pairs:
        sen1,sen2=sen_pairs
        for word in sen1:
            all_words.append(word)
        for word in sen2:
            all_words.append(word)
    counter=collections.Counter(all_words)
    sorted_list=sorted(counter.items(),key=lambda x:x[1],reverse=True)
    for word,freq in sorted_list:
        word2id[word]=len(word2id)
    return word2id

class Dataset:
    def __init__(self,sentence_pairs,word2id,mode="train",label_list=None):
        self.sentence_pairs=sentence_pairs
        self.mode=mode
        self.label_list=label_list
        self.sample_nums=len(self.sentence_pairs)
        self.indicator=0
        self.word2id=word2id
        self.PAD_ID=self.word2id["PAD"]
        self.max_2_length=30
        self.max_1_length=30
        self.process_pairs()

    def process_pairs(self):
        self.sentence1=[]
        self.sentence2=[]
        for sentence_pair in self.sentence_pairs:
            assert len(sentence_pair)==2
            self.sentence1.append(sentence_pair[0])
            self.sentence2.append(sentence_pair[1])

        if self.mode=="train":
            assert self.label_list!=None
            assert len(self.label_list)==len(self.sentence1)==len(self.sentence2)==self.sample_nums
            self.label_list=np.array(self.label_list)
        self.sentence1=np.array(self.sentence1)
        self.sentence2=np.array(self.sentence2)

    def pad_sentence_pairs(self,batch_sentence1,batch_sentence2):
        pad_sentence1=[]
        pad_sentence2=[]
        max_1_length=self.max_1_length
        max_2_length=self.max_2_length
        assert len(batch_sentence1)==len(batch_sentence2)
        features=[]
        for sen1,sen2 in zip(batch_sentence1,batch_sentence2):
            sen1_len=len(sen1)
            sen2_len=len(sen2)
            if sen1_len>max_1_length:
                sen1_len=max_1_length
            if sen2_len>max_2_length:
                sen2_len=max_2_length

            features.append([sen1_len,sen2_len])

        for i,sen in enumerate(batch_sentence1):
            if len(sen)>=max_1_length:
                pad_sentence1.append(sen[:max_1_length])
            else:
                pad_sentence1.append(sen[:max_1_length]+[self.PAD_ID]*(max_1_length-len(sen)))

        for i,sen in enumerate(batch_sentence2):
            if len(sen)>=max_2_length:
                pad_sentence2.append(sen[:max_2_length])
            else:
                pad_sentence2.append(sen[:max_2_length]+[self.PAD_ID]*(max_2_length-len(sen)))
        return np.array(pad_sentence1),np.array(pad_sentence2),np.array(features)	

    def shuffle_fn(self):
        assert self.mode=="train"
        shuffle_index=np.random.permutation(self.sample_nums)
        self.sentence1=self.sentence1[shuffle_index]
        self.sentence2=self.sentence2[shuffle_index]
        self.sentence1_length=self.sentence1_length[shuffle_index]
        self.sentence2_length=self.sentence2_length[shuffle_index]
        self.label_list=self.label_list[shuffle_index]

    def next_batch(self,batch_size):
        end_indicator=self.indicator+batch_size
        if end_indicator>self.sample_nums:
            self.indicator=0
            end_indicator=batch_size
            if self.mode=="train":
                self.shuffle_fn()
        batch_sentence1=self.sentence1[self.indicator:end_indicator]
        batch_sentence2=self.sentence2[self.indicator:end_indicator]
        batches_data=self.pad_sentence_pairs(batch_sentence1,batch_sentence2)
        if self.mode=="train":
            batch_label_list=self.label_list[self.indicator:end_indicator]
        self.indicator+=batch_size
        if self.mode=="train":
            batch_x1,batch_x2,batch_features=batches_data
            return batch_x1,batch_x2,batch_label_list,batch_features
        else:
            batch_x1,batch_x2,batch_features=batches_data
            return batch_x1,batch_x2,batch_features
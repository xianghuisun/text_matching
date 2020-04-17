import os
import numpy as np
import pandas as pd
import random
import pickle

# def load_data(data_path):
# 	with open(data_path) as f:
# 		lines=f.readlines()
# 	print(len(lines))
# 	sentence_pairs=[]
# 	label_list=[]
# 	for line in lines:
# 		line_split=line.strip().split("\t")
# 		sentence1=[number for number in line_split[0].strip().split()]
# 		sentence2=[number for number in line_split[1].strip().split()]
# 		sentence_pairs.append((sentence1,sentence2))
# 		if len(line_split)==3:
# 			label_list.append(int(line_split[-1]))
# 	if len(line_split)==3:
# 		return sentence_pairs,label_list
# 	else:
# 		return sentence_pairs

def process_sentence(sentences):
    import re
    sentences=re.sub("([?<>,.!@#$%&*;:])",repl=" ",string=sentences)
    sentences=re.sub("[' ']+",repl=' ',string=sentences)
    return sentences.strip().split()

def load_data(data_path="../quora_duplicate_questions.tsv",need_process=True):
    with open(data_path) as f:
        lines=f.readlines()
    sentence_pairs=[]
    label_list=[]
    i=0
    for line in lines[1:]:
        line_split=line.strip().split("\t")
        try:
            assert len(line_split)==6
        except:
            #print(line_split)
            i+=1
            continue
        label=line_split[-1]
        if need_process:
            seq_1=process_sentence(line_split[-2])
            seq_2=process_sentence(line_split[-3])
        else:
            seq_1=line_split[-2].strip().split()
            seq_2=line_split[-3].strip().split()

        sentence_pairs.append((seq_1,seq_2))
        label_list.append(label)
    print(i)
    return sentence_pairs,label_list


def get_parameter(train_sentence_pairs,word2vec_model=None,embed_dim=300):
	word2id={}
	all_words=[]
	for sen_pairs in train_sentence_pairs:
		sen1,sen2=sen_pairs
		for word in sen1:
			all_words.append(word)
		for word in sen2:
			all_words.append(word)
	import collections
	counter=collections.Counter(all_words)
	sorted_words=sorted(counter.items(),key=lambda x:x[1],reverse=True)
	words_list=[]
	for word,freq in sorted_words:
		words_list.append(word)

	print("There are %d unique words in train sen_pairs"%len(words_list))
	if word2vec_model!=None:
		words_in_word2vec=list(word2vec_model.wv.vocab.keys())
		print("There are %d unique words in word2vec model " %len(words_in_word2vec))
		embedding_matrix=[]
		for i,word in enumerate(words_in_word2vec):
			if i==0:
				word2id["--UNK--"]=len(word2id)
				embedding_matrix.append(np.random.uniform(-0.5,0.5,word2vec_model.vector_size))
				word2id["--PAD--"]=len(word2id)
				embedding_matrix.append(np.zeros(word2vec_model.vector_size))
			if word in words_list:
				word2id[word]=len(word2id)
				embedding_matrix.append(word2vec_model[word])

		embedding_matrix=np.array(embedding_matrix)
	else:
		word2id["--UNK--"]=len(word2id)
		word2id["--PAD--"]=len(word2id)
		for word in words_list:
			word2id[word]=len(word2id)
		embedding_matrix=np.random.randn(len(word2id),embed_dim)
	return word2id,embedding_matrix

class Dataset:
	def __init__(self,sentence_pairs,word2id,mode="train",label_list=None):
		self.sentence_pairs=sentence_pairs
		self.mode=mode
		if self.mode=="train":
			self.label_list=np.array(label_list)
		self.sample_nums=len(self.sentence_pairs)
		self.indicator=0
		self.word2id=word2id
		self.vocab_size=len(self.word2id)
		self.PAD_ID=word2id["--PAD--"]
		self.max_seq_length=30
		self.sentences_pairs_to_id()

	def sentences_pairs_to_id(self):
		self.sentence1=[]
		self.sentence2=[]
		for sen_pairs in self.sentence_pairs:
			sen1,sen2=sen_pairs
			sen1_id_list=[]
			sen2_id_list=[]
			for word in sen1:
				sen1_id_list.append(self.word2id.get(word,self.word2id["--UNK--"]))
			self.sentence1.append(sen1_id_list)
			for word in sen2:
				sen2_id_list.append(self.word2id.get(word,self.word2id["--UNK--"]))
			self.sentence2.append(sen2_id_list)



	def pad_sentence_pairs(self,batch_sentence1,batch_sentence2):
		batch_1_length=[len(sen) for sen in batch_sentence1]
		batch_2_length=[len(sen) for sen in batch_sentence2]
		max_1_length=self.max_seq_length
		max_2_length=self.max_seq_length
		pad_sentence1=[]
		pad_sentence2=[]
		new_1_length=[]
		new_2_length=[]
		for i,sen in enumerate(batch_sentence1):
			assert len(sen)==batch_1_length[i]
			if len(sen)>=max_1_length:
				new_1_length.append(max_1_length)
				pad_sentence1.append(sen[:max_1_length])
			else:
				new_1_length.append(len(sen))
				pad_sentence1.append(sen[:max_1_length]+[self.PAD_ID]*(max_1_length-len(sen)))

		for i,sen in enumerate(batch_sentence2):
			assert len(sen)==batch_2_length[i]
			if len(sen)>=max_2_length:
				new_2_length.append(max_2_length)
				pad_sentence2.append(sen[:max_2_length])
			else:
				new_2_length.append(len(sen))
				pad_sentence2.append(sen[:max_2_length]+[self.PAD_ID]*(max_2_length-len(sen)))
		return np.array(pad_sentence1),np.array(pad_sentence2),np.array(new_1_length),np.array(new_2_length)		

	def shuffle_fn(self):
		assert self.mode=="train"
		shuffle_index=np.random.permutation(self.sample_nums)
		self.sentence1=np.array(self.sentence1)[shuffle_index]
		self.sentence2=np.array(self.sentence2)[shuffle_index]
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
			return batches_data,batch_label_list
		else:
			return batches_data

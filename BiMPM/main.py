from data_process import *
from matching import *
import pickle

train_data_path="../train.txt"
test_data_path="../test.txt"

sentence_pairs,label_list=load_data(train_data_path)
valid_sample_nums=20000
#test_sample_nums=20000
train_sample_nums=len(sentence_pairs)-valid_sample_nums

train_sentence_pairs=sentence_pairs[:train_sample_nums]
train_label_list=label_list[:train_sample_nums]

valid_sentence_pairs=sentence_pairs[-valid_sample_nums:]
valid_label_list=label_list[-valid_sample_nums:]
test_sentence_pairs=load_data(test_data_path)

#test_sentence_pairs=sentence_pairs[]
print("Print the train,valid,test numbers:---------------------------:")
print("train sample number are-------> ",len(train_sentence_pairs),len(train_label_list))
print("valid sample numbers are ---------->",len(valid_sentence_pairs),len(valid_label_list))
print("test sample numbers are--------------->",len(test_sentence_pairs))
print("---------------------------------------------------------------")

import os
import gensim
word2vec_model=gensim.models.word2vec.Word2Vec.load("/home/aistudio/work/word2vec_model")
if os.path.exists("./parameter.pkl"):
    with open("./parameter.pkl","rb") as f:
        word2id,embedding_matrix=pickle.load(f)
else:
    with open("./parameter.pkl","wb") as f:
        word2id,embedding_matrix=get_parameter(train_sentence_pairs,word2vec_model)
        parameter=(word2id,embedding_matrix)
        pickle.dump(parameter,f)


vocab_size=len(word2id)
print("vocab size is ",vocab_size)

dataset=Dataset(sentence_pairs=train_sentence_pairs,word2id=word2id,mode="train",label_list=train_label_list)
test_dataset=Dataset(sentence_pairs=test_sentence_pairs,word2id=word2id,mode="test",label_list=None)
valid_dataset=Dataset(sentence_pairs=valid_sentence_pairs,word2id=word2id,mode="train",label_list=valid_label_list)


class Args:
    def __init__(self,vocab_size):
        self.embed_dim=word2vec_model.vector_size
        self.num_classes=2
        self.learning_rate=0.001
        self.epochs=50
        self.batch_size=256
        self.embed_size=vocab_size
        self.context_hidden_dim=128
        self.compose_hidden_dim=100


args=Args(vocab_size)

model=Model(args.embed_size,embed_matrix=embedding_matrix,embed_dim=args.embed_dim)

def evaluate(sess):
    num_batches=valid_dataset.sample_nums//args.batch_size
    total_loss=0.0
    total_acc=0.0
    for i in range(num_batches):
        batches_data,batch_label=valid_dataset.next_batch(batch_size=args.batch_size)
        assert len(batches_data)==4
        feed_dict={model.premise:batches_data[0],model.hypothesis:batches_data[1],
                model.premise_length:batches_data[2],model.hypothesis_length:batches_data[3],
                model.y:batch_label,model.keep_prob:1.0}
        batch_loss,batch_acc=sess.run([model.loss,model.accuracy],feed_dict=feed_dict)
        total_acc+=batch_acc
        total_loss+=batch_loss
    return total_loss/num_batches,total_acc/num_batches

def test(sess):
    num_batches=test_dataset.sample_nums//100
    f=open("./xhsun_predict.txt","w")
    for i in range(num_batches):
        batches_data=test_dataset.next_batch(100)
        #assert len(batches_data)==4
        feed_dict={model.premise:batches_data[0],model.hypothesis:batches_data[1],
                model.premise_length:batches_data[2],model.hypothesis_length:batches_data[3],model.keep_prob:1.0}
        predict=sess.run(model.prediction,feed_dict=feed_dict)
        for id_ in predict:
            f.write(str(id_))
            f.write("\n")
    f.close()
    with open("./xhsun_predict.txt","r") as f:
        lines=f.readlines()
    assert len(lines)==test_dataset.sample_nums
    print("Has saved the predict result in current folder!")

def train():
    num_batches=dataset.sample_nums//args.batch_size
    saved_acc=0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver(max_to_keep=3)
        for epoch in range(args.epochs):
            
            for i in range(num_batches):
                batches_data,batch_label=valid_dataset.next_batch(batch_size=args.batch_size)
                assert len(batches_data)==4
                feed_dict={model.premise:batches_data[0],model.hypothesis:batches_data[1],
                        model.premise_length:batches_data[2],model.hypothesis_length:batches_data[3],
                        model.y:batch_label,model.keep_prob:0.8}
                batch_loss,batch_acc,_=sess.run([model.loss,model.accuracy,model.train_op],feed_dict=feed_dict)
                if i%200==0:
                    print("Epoch is %d,loss value is %f and accuracy is %f " % (epoch,batch_loss,batch_acc))
            valid_loss,valid_acc=evaluate(sess)
            print("Epoch is %d and valid accuracy is %f " %(epoch,valid_acc))
            if valid_acc>saved_acc:
                test(sess)
                saver.save(sess,"./model.ckpt")
                saved_acc=valid_acc
                print("Has test test dataset once and saved model in model.ckpt")
                if valid_acc>=0.998:
                    return

train()


                


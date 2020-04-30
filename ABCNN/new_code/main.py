from data_process import *
from model import *
import pickle

# train_data_path="../train.txt"
# test_data_path="../test.txt"

# sentence_pairs,label_list=load_data(train_data_path)
# test_sentence_pairs=load_data(test_data_path)
# valid_sample_nums=20000
# #test_sample_nums=20000
# train_sample_nums=len(sentence_pairs)-valid_sample_nums

# train_sentence_pairs=sentence_pairs[:train_sample_nums]
# train_label_list=label_list[:train_sample_nums]

# valid_sentence_pairs=sentence_pairs[-valid_sample_nums:]
# valid_label_list=label_list[-valid_sample_nums:]

# print("Print the train,valid,test numbers:---------------------------:")
# print("train sample number are-------> ",len(train_sentence_pairs),len(train_label_list))
# print("valid sample numbers are ---------->",len(valid_sentence_pairs),len(valid_label_list))
# print("test sample numbers are--------------->",len(test_sentence_pairs))
# print("---------------------------------------------------------------")

# import os
# import gensim
# word2vec_model=gensim.models.word2vec.Word2Vec.load("/home/aistudio/work/word2vec_model")
# if os.path.exists("./parameter.pkl"):
#     with open("./parameter.pkl","rb") as f:
#         word2id,embedding_matrix=pickle.load(f)
# else:
#     with open("./parameter.pkl","wb") as f:
#         word2id,embedding_matrix=get_parameter(train_sentence_pairs)
#         parameter=(word2id,embedding_matrix)
#         pickle.dump(parameter,f)


sentence_pairs,label_list=load_data()

valid_samples=10000
test_samples=10000
train_samples=len(sentence_pairs)-test_samples-valid_samples#384000

train_sentence_pairs=sentence_pairs[:train_samples]
train_label_list=label_list[:train_samples]

valid_sentence_pairs=sentence_pairs[train_samples:train_samples+valid_samples]
valid_label_list=label_list[train_samples:train_samples+valid_samples]

test_sentence_pairs=sentence_pairs[-test_samples:]
test_label_list=label_list[-test_samples:]
print(len(train_sentence_pairs),len(valid_sentence_pairs),len(test_sentence_pairs))

if os.path.exists("./parameter.pkl"):
    with open("./parameter.pkl","rb") as f:
        word2id,embedding_matrix=pickle.load(f)
else:
    word2id,embedding_matrix=get_parameter(train_sentence_pairs)
    with open("./parameter.pkl","wb") as f: 
        pickle.dump((word2id,embedding_matrix),f)

train_dataset=Dataset(train_sentence_pairs,word2id,label_list=train_label_list)
valid_dataset=Dataset(valid_sentence_pairs,word2id,mode="valid",label_list=valid_label_list)
test_dataset=Dataset(test_sentence_pairs,word2id,mode="test")



vocab_size=len(word2id)
print("vocab size is ",vocab_size)

# train_dataset=Dataset(sentence_pairs=train_sentence_pairs,word2id=word2id,mode="train",label_list=train_label_list)
# test_train_dataset=Dataset(sentence_pairs=test_sentence_pairs,word2id=word2id,mode="test",label_list=None)
# valid_train_dataset=Dataset(sentence_pairs=valid_sentence_pairs,word2id=word2id,mode="train",label_list=valid_label_list)


class Args:
    def __init__(self,vocab_size,seq_length_p,seq_length_h):
        self.seq_length_p=seq_length_p
        self.seq_length_h=seq_length_h
        self.filter_width=3
        #self.embedding_dim=word2vec_model.vector_size
        #self.filter_height=self.embedding_dim
        self.cnn1_filters=50
        self.cnn2_filters=50
        self.num_classes=2
        self.learning_rate=0.005
        self.epochs=50
        self.batch_size=100
        self.vocab_size=vocab_size
        self.fc_dim=256
args=Args(vocab_size,seq_length_p=25,seq_length_h=25)

model=Model(embedding_matrix,args.num_classes,args.seq_length_p,args.seq_length_h)
#model.forward()

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

# def test(sess):
#     num_batches=test_dataset.sample_nums//args.batch_size
#     f=open("./xhsun_predict.txt","w")
#     for i in range(num_batches):
#         batches_data=test_train_dataset.next_batch(100)
#         assert len(batches_data)==4
#         feed_dict={model.premise:batches_data[0],model.hypothesis:batches_data[1],
#                 model.premise_length:batches_data[2],model.hypothesis_length:batches_data[3]}
#         predict=sess.run(model.prediction,feed_dict=feed_dict)
#         for id_ in predict:
#             f.write(str(id_))
#             f.write("\n")
#     f.close()
#     with open("./xhsun_predict.txt","r") as f:
#         lines=f.readlines()
#     assert len(lines)==test_train_dataset.sample_nums
#     print("Has saved the predict result in current folder!")

def test(sess):
    num_batches=test_dataset.sample_nums//args.batch_size
    predict_list=[]
    for i in range(num_batches):
        batches_data=test_dataset.next_batch(args.batch_size)
        assert len(batches_data)==4
        feed_dict={model.premise:batches_data[0],model.hypothesis:batches_data[1],
                model.premise_length:batches_data[2],model.hypothesis_length:batches_data[3],model.keep_prob:1.0}
        predict=sess.run(model.prediction,feed_dict=feed_dict)
        for i in list(predict):
            predict_list.append(i)

    assert len(predict_list)==len(test_label_list)==test_dataset.sample_nums
    correct=0
    for i,j in zip(predict_list,test_label_list):
        j=int(j)
        assert type(i)==int==type(j)
        if i==j:
            correct+=1
    print("Test result accuracy is ",correct/test_dataset.sample_nums)

def train():
    num_batches=train_dataset.sample_nums//args.batch_size
    saved_acc=0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver(max_to_keep=3)
        for epoch in range(args.epochs):
            
            for i in range(num_batches):
                batches_data,batch_label=train_dataset.next_batch(batch_size=args.batch_size)
                assert len(batches_data)==4
                feed_dict={model.premise:batches_data[0],model.hypothesis:batches_data[1],
                        model.premise_length:batches_data[2],model.hypothesis_length:batches_data[3],
                        model.y:batch_label,model.keep_prob:0.75}
                batch_loss,batch_acc,_=sess.run([model.loss,model.accuracy,model.train_op],feed_dict=feed_dict)
                if i%300==0:
                    print("Epoch is %d,loss value is %f and accuracy is %f " % (epoch,batch_loss,batch_acc))
            valid_loss,valid_acc=evaluate(sess)
            print("Epoch is %d and valid accuracy is %f " %(epoch,valid_acc))
            if valid_acc>saved_acc:
                saver.save(sess,"./model.ckpt")
                saved_acc=valid_acc
                test(sess)
                print("Has test test train_dataset once and saved model in model.ckpt")
                if valid_acc>=0.998:
                    test(sess)
                    return
        test(sess)

train()


                


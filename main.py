from data_process import *
from model import *


train_data_path="./train.txt"
test_data_path="./test.txt"

class Config:
    def __init__(self,vocab_size):
        self.num_classes=2
        self.embedding_dim=512
        self.vocab_size=vocab_size
        self.batch_size=128
        self.learning_rate=0.001
        self.keep_prob=0.5

sentence_pairs,label_list=load_data(data_path=train_data_path)
#valid_sample_nums=len(sentence_pairs)-len(sentence_pairs)*4//5
train_sample_nums=len(sentence_pairs)

train_sentence_pairs=sentence_pairs[:train_sample_nums]
train_label_list=label_list[:train_sample_nums]
print(len(train_sentence_pairs),len(train_label_list))
#valid_sentence_pairs=sentence_pairs[train_sample_nums:]
#valid_label_list=label_list[train_sample_nums:]

#print(len(train_sentence_pairs),len(valid_sentence_pairs),len(train_label_list),len(valid_label_list))

test_sentence_pairs=load_data(data_path=test_data_path)

word2id=get_word2id(train_sentence_pairs)
vocab_size=len(word2id)
print("vocab size is ",vocab_size)


class Args:
    def __init__(self,vocab_size,seq_length_p,seq_length_h):
        self.seq_length_p=seq_length_p
        self.seq_length_h=seq_length_h
        self.filter_width=3
        self.embedding_dim=512
        self.filter_height=self.embedding_dim
        self.cnn1_filters=100
        self.cnn2_filters=100
        self.num_classes=2
        self.learning_rate=0.001
        self.epochs=100
        self.batch_size=128
        self.vocab_size=vocab_size
        self.fc_dim=64
args=Args(vocab_size,seq_length_p=30,seq_length_h=30)




config=Config(vocab_size)
model=Model(embed_dim=args.embedding_dim,embed_size=vocab_size,num_classes=args.num_classes,
        seq_length_p=args.seq_length_p,seq_length_h=args.seq_length_h)

dataset=Dataset(sentence_pairs=train_sentence_pairs,word2id=word2id,mode="train",label_list=train_label_list)
#valid_dataset=Dataset(sentence_pairs=valid_sentence_pairs,
#	PAD_ID=vocab_size,mode="train",label_list=valid_label_list)
test_dataset=Dataset(sentence_pairs=test_sentence_pairs,word2id=word2id,mode="test",label_list=None)

epochs=args.epochs
def train_batch(sess,feed_dict):
    loss_val,_,acc_val=sess.run([model.loss,model.train_op,model.accuracy],feed_dict=feed_dict)
    return loss_val,acc_val

def evaluate(sess):
    num_batch=valid_dataset.sample_nums//config.batch_size
    total_loss=0.0
    total_acc=0.0
    for i in range(num_batch):
        batches=valid_dataset.next_batch(batch_size=config.batch_size)

        feed_dict={model.premise:batches[0][0],model.hypothesis:batches[0][1],
                    model.y:batches[1]}
        loss_val,acc_val=train_batch(sess,feed_dict)
        total_loss+=loss_val
        total_acc+=acc_val
    return total_loss/num_batch,total_acc/num_batch

def test(sess):
    num_batches_=test_dataset.sample_nums//100
    result=[]
    for i in range(num_batches_):
        batches=test_dataset.next_batch(batch_size=100)
        feed_dict={model.premise:batches[0],model.hypothesis:batches[1]}
        result.extend(list(sess.run(model.prediction,feed_dict)))#(batch_size)
    assert len(result)==test_dataset.sample_nums
    with open("./xhsun_predict.txt","w") as f:
        for i in result:
            f.write(str(i))
            f.write("\n")



def train():
    num_batches=dataset.sample_nums//config.batch_size
    saver=tf.train.Saver(max_to_keep=3)
    save_acc=0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss=0.0
            epoch_acc=0.0
            for i in range(num_batches):
                batches=dataset.next_batch(batch_size=config.batch_size)
                #assert len(batches)==2 and len(batches[0])==4
                feed_dict={model.premise:batches[0],model.hypothesis:batches[1],
                            model.y:batches[2]}
                loss_val,acc_val=train_batch(sess,feed_dict)
                epoch_loss+=loss_val
                epoch_acc+=acc_val
                if i%200==0:
                    print("Epoch is %d, loss_val is %f,acc_val is %f "%(epoch,loss_val,acc_val))
            #valid_loss,valid_acc=evaluate(sess)
            epoch_loss/=num_batches
            epoch_acc/=num_batches
            print("*"*100)
            print("Epoch is %d , epoch loss is %f and epoch accuracy value is %f " % (epoch,epoch_loss,epoch_acc))
            print("*"*100)
            if epoch_acc>=0.998:
                saver.save(sess,"./textmatching.ckpt")
                print("model accuracy in train dataset has over 0.998")
                test(sess)
                return
            #print("Epoch is %d valid loss value is %f and valid accuracy is %f " % (epoch,valid_loss,valid_acc))
            if save_acc<epoch_acc:
                save_acc=epoch_acc
                saver.save(sess,"./textmatching.ckpt")
                test(sess)
                print("Has saved model in textmatching.ckpt")

            print("One epoch has trained over!")

train()
print("It's over!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")

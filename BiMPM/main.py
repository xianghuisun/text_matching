from data_process import *
from model import *
import pickle

train_data_path="./snli_1.0/snli_1.0_train.txt"
test_data_path="./snli_1.0/snli_1.0_test.txt"
valid_data_path="./snli_1.0/snli_1.0_dev.txt"

train_sentence_pairs,train_label_list=load_data(train_data_path)
valid_sentence_pairs,valid_label_list=load_data(valid_data_path)
test_sentence_pairs,test_label_list=load_data(test_data_path)

print(len(train_sentence_pairs),len(valid_sentence_pairs),len(test_sentence_pairs))

if os.path.exists("./parameter.pkl"):
    with open("./parameter.pkl","rb") as f:
        word2id,label2id,embedding_matrix=pickle.load(f)
else:
    word2id,label2id,embedding_matrix=get_parameter(train_sentence_pairs,train_label_list)
    with open("./parameter.pkl","wb") as f: 
        pickle.dump((word2id,label2id,embedding_matrix),f)

train_dataset=Dataset(train_sentence_pairs,word2id,label2id,label_list=train_label_list)
valid_dataset=Dataset(valid_sentence_pairs,word2id,label2id,mode="valid",label_list=valid_label_list)
test_dataset=Dataset(test_sentence_pairs,word2id,label2id,mode="test",label_list=test_label_list)
test_label_list=test_dataset.label_list


vocab_size=len(word2id)
print("vocab size is ",vocab_size)
print(label2id)

class Args:
    def __init__(self,vocab_size):

        #self.embedding_dim=word2vec_model.vector_size
        self.num_classes=len(label2id)
        self.learning_rate=0.005
        self.epochs=50
        self.batch_size=100
        self.vocab_size=vocab_size
        self.fc_dim=512
        self.context_hidden_dim=600
        self.compose_hidden_dim=600
args=Args(vocab_size)

model=Model(args,embedding_matrix)
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

def test(sess):
    num_batches=test_dataset.sample_nums//args.batch_size
    rest_nums=test_dataset.sample_nums-(num_batches*args.batch_size)
    print("Rest number is ",rest_nums)

    predict_list=[]
    for i in range(num_batches):
        batches_data=test_dataset.next_batch(args.batch_size)
        assert len(batches_data)==4
        feed_dict={model.premise:batches_data[0],model.hypothesis:batches_data[1],
                model.premise_length:batches_data[2],model.hypothesis_length:batches_data[3],model.keep_prob:1.0}
        predict=sess.run(model.prediction,feed_dict=feed_dict)
        for j in list(predict):
            predict_list.append(j)
        if i==num_batches-1:
            batches_data=test_dataset.next_batch(rest_nums)
            assert len(batches_data)==4
            feed_dict={model.premise:batches_data[0],model.hypothesis:batches_data[1],
                    model.premise_length:batches_data[2],model.hypothesis_length:batches_data[3],model.keep_prob:1.0}
            predict=sess.run(model.prediction,feed_dict=feed_dict)
            print("predict .shape ")
            for j in list(predict):
                predict_list.append(j)

    assert len(predict_list)==len(test_label_list)==test_dataset.sample_nums
    correct=0
    for i,j in zip(predict_list,test_label_list):
        j=int(j)
        i=int(i)
        assert type(i)==int==type(j)
        if i==j:
            correct+=1
    print("Test result accuracy is ",correct/test_dataset.sample_nums)


config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.8
def train():
    num_batches=train_dataset.sample_nums//args.batch_size
    saved_acc=0.0
    with tf.Session(config=config) as sess:
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


                


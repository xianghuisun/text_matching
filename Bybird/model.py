from ABCNN3 import ABCNN
from EMSI import EMSI_model

def focal_loss_fn(y_true,y_pred,gamma=2,alpha=0.6):
    cross_entroy0=tf.multiply(y_true,-tf.log(y_pred))
    cross_entroy1=tf.multiply(tf.subtract(1.,y_true),-tf.log(tf.subtract(1.,y_pred)))

    fl_0=tf.pow(tf.subtract(1.,y_pred),gamma)*alpha*cross_entroy0
    fl_1=(1-alpha)*tf.pow(y_pred,gamma)*cross_entroy1
    return tf.reduce_mean(tf.add(fl_0,fl_1))

class Bybird_Model:
	def __init__(self,num_classes,embedding_matrix,seq_p,seq_h):
		self.abcnn_model=ABCNN()
		self.emsi=EMSI_model()
		self.embedding_matrix=embedding_matrix
		self.embed_dim=int(embedding_matrix.shape[-1])
		self.seq_p=seq_p
		self.seq_h=seq_h
		self.num_classes=num_classes
		self.add_placeholder()

	def add_placeholder(self):
		self.premise=tf.placeholder(shape=[None,self.seq_p],dtype=tf.int32)
		self.hypothesis=tf.placeholder(shape=[None,self.seq_h],dtype=tf.int32)
		self.premise_length=tf.placeholder(shape=[None],dtype=tf.int32)
		self.hypothesis_length=tf.placeholder(shape=[None],dtype=tf.int32)
		self.y=tf.placeholder(shape=[None],dtype=tf.int32)

	def embedding_layer(self):
		embedding_matrix=tf.Variable(self.embedding_matrix,dtype=tf.float32)
		self.p_embedding=tf.nn.embedding_lookup(params=embedding_matrix,ids=self.premise)
		self.h_embedding=tf.nn.embedding_lookup(params=embedding_matrix,ids=self.hypothesis)

	def forward(self):
		self.embedding_layer()
		abcnn_out_p,abcnn_out_h=self.abcnn_model.forward(self.p_embedding,self.h_embedding)
		#(batch_size,seq_length,50)
		emsi_output=self.emsi.forward(abcnn_out_p,abcnn_out_h,self.premise_length,self.hypothesis_length)
		#(batch_size,64)
		self.logits=tf.layers.dense(emsi_output,units=self.num_classes)#(batch_size,2)
		self.train()

	def train(self):
		labels=tf.one_hot(self.y,self.num_classes)
		self.loss=focal_loss_fn(y_pred=tf.nn.sigmoid(self.logits),y_true=labels)
		optimizer=tf.train.AdamOptimizer(0.001)
		grads=tf.gradients(self.loss,tf.trainable_variables())
		grads,_=tf.clip_by_global_norm(grads,2.0)
		self.train_op=optimizer.apply_gradients(zip(grads,tf.trainable_variables()))
		self.prediction=tf.cast(tf.argmax(tf.nn.softmax(self.logits),axis=-1),dtype=tf.int32)
		self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.prediction,self.y),tf.float32))



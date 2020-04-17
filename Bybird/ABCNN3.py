import tensorflow as tf
import numpy as np

class ABCNN:
	def __init__(self,filter_width=3,filters=50):
		self.filter_width=filter_width
		self.filters=filters

	def get_euclidean(self,x1,x2):
		#(batch_size,seq_1,dim,1),(batch_size,seq_2,dim,1)
		assert x1.shape[-1]==x2.shape[-1]==1 and x1.shape[2]==x2.shape[2]#features
		x2_t=tf.transpose(x2,perm=[0,3,2,1])#(batch_size,1,dim,seq_2)
		euclidean_distance=tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x1,x2_t)),axis=2))#(batch_size,seq_1,seq_2)
		assert euclidean_distance.shape[1]==x1.shape[1] and euclidean_distance.shape[2]==x2.shape[1]
		return tf.exp(-euclidean_distance/2.0)#nan situation

	def abcnn1(self,x1,x2):
		#(batch_size,seq_1,embedding_dim,1)(batch_size,seq_2,embeddng_dim,1)
		seq_1=x1.shape[1]
		seq_2=x2.shape[1]
		features=x1.shape[2]
		assert features==x2.shape[2]
		attention_matrix=self.get_euclidean(x1,x2)#(batch_size,seq_1,seq_2)
		attention_matrix_t=tf.transpose(attention_matrix,perm=[0,2,1])#(batch_size,seq_2,seq_1)
		with tf.variable_scope("abcnn1_W",reuse=tf.AUTO_REUSE):
			w1=tf.get_variable(name="w1",shape=[seq_2,features],initializer=tf.random_uniform_initializer(-1.0,1.0))
			w2=tf.get_variable(name="w2",shape=[seq_1,features],initializer=tf.random_uniform_initializer(-1.0,1.0))
		x1_attention=tf.einsum("ijk,kl->ijl",attention_matrix,w1)#(batch_size,seq_1,features)
		x2_attention=tf.einsum("ijk,kl->ijl",attention_matrix_t,w2)#(batch_size,seq_2,features)
		x1_attention=tf.expand_dims(x1_attention,axis=-1)
		x2_attention=tf.expand_dims(x2_attention,axis=-1)
		x1_concat=tf.concat(values=[x1_attention,x1,tf.subtract(x1,x1_attention),x1*x1_attention],axis=-1)
		x2_concat=tf.concat(values=[x2_attention,x2,tf.subtract(x2,x2_attention),x2*x2_attention],axis=-1)
		#(batch_size,seq_,features,4)
		return x1_concat,x2_concat

	def wide_conv(self,x1,x2):
		#(batch_size,seq_,features,4)
		#return (batch_size,seq_+2,filters,1)
		in_channels=int(x1.shape[-1])
		#assert in_channels==x2.shape[-1]
		wide_x1=tf.pad(x1,[[0,0],[self.filter_width-1,self.filter_width-1],[0,0],[0,0]])
		wide_x2=tf.pad(x2,[[0,0],[self.filter_width-1,self.filter_width-1],[0,0],[0,0]])
		weights1=tf.Variable(tf.random_normal(shape=[self.filter_width,int(x1.shape[2]),in_channels,self.filters],dtype=tf.float32))
		conv_x1=tf.nn.conv2d(wide_x1,filter=weights1,strides=[1,1,1,1],padding="VALID")
		weights2=tf.Variable(tf.random_normal(shape=[self.filter_width,int(x1.shape[2]),in_channels,self.filters],dtype=tf.float32))
		conv_x2=tf.nn.conv2d(wide_x2,filter=weights2,strides=[1,1,1,1],padding="VALID")
		conv_x1=tf.transpose(conv_x1,perm=[0,1,3,2])
		conv_x2=tf.transpose(conv_x2,perm=[0,1,3,2])
		assert conv_x1.shape[-1]==conv_x2.shape[-1]==1
		return conv_x1,conv_x2


	def abcnn2(self,x1,x2):
		#(batch_size,seq_+2,filters,1)
		attention_matrix=self.get_euclidean(x1,x2)#(batch_size,seq_1+2,seq_2+2)
		x1_attention=tf.reduce_sum(attention_matrix,axis=2)#(batch_size,seq_1+2)
		x2_attention=tf.reduce_sum(attention_matrix,axis=1)#(batch_size,seq_2+2)
		x1_attention=tf.expand_dims(tf.expand_dims(x1_attention,axis=-1),axis=-1)
		x2_attention=tf.expand_dims(tf.expand_dims(x2_attention,axis=-1),axis=-1)
		pool_x1_list=[]
		pool_x2_list=[]
		assert x1.shape[2]==x2.shape[2] and x1.shape[-1]==1==x2.shape[-1]
		x1_length=x1.shape[1]-self.filter_width+1
		x2_length=x2.shape[1]-self.filter_width+1
		for i in range(x1_length):
			pool_x1_list.append(tf.reduce_mean(x1[:,i:i+self.filter_width,:,:]*x1_attention[:,i:i+self.filter_width,:,:],
				axis=1,keep_dims=True))#(batch_size,1,filters,1)
		x1_pool=tf.concat(pool_x1_list,axis=1)#(batch_size,seq_1,filters,1)
		x1_pool=tf.reshape(x1_pool,shape=[-1,x1_length,x1.shape[2]])
		for i in range(x2_length):
			pool_x2_list.append(tf.reduce_mean(x2[:,i:i+self.filter_width,:,:]*x2_attention[:,i:i+self.filter_width,:,:],
				axis=1,keep_dims=True))#(batch_size,1,1,1)
		x2_pool=tf.concat(pool_x2_list,axis=1)
		x2_pool=tf.reshape(x2_pool,shape=[-1,x2_length,x2.shape[2]])
		return x1_pool,x2_pool

	def forward(self,premise,hypothesis):
		#(batch_size,seq_p,dim)(batch_size,seq_h,dim)
		x1=tf.expand_dims(premise,axis=-1)
		x2=tf.expand_dims(hypothesis,axis=-1)
		abcnn1_x1,abcnn1_x2=self.abcnn1(x1,x2)
		wide_x1,wide_x2=self.wide_conv(abcnn1_x1,abcnn1_x2)
		abcnn3_x1,abcnn3_x2=self.abcnn2(wide_x1,wide_x2)
		#print(premise.shape,abcnn3_x1.shape,"***********",hypothesis.shape,abcnn3_x2.shape)
		return abcnn3_x1,abcnn3_x2
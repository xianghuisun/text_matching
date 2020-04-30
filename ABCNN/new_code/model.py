import tensorflow as tf

class Model:
	def __init__(self,embedding_matrix,num_classes,seq_length_p,seq_length_h,filter_width=3):
		self.seq_length_h=seq_length_h
		self.seq_length_p=seq_length_p
		self.embed_dim=embedding_matrix.shape[1]
		self.embed_size=embedding_matrix.shape[0]
		self.embedding_matrix=tf.Variable(embedding_matrix,dtype=tf.float32)
		self.num_classes=num_classes
		self.filter_width=filter_width
		self.conv_filters=50
		self.abcnn1=True
		self.abcnn2=True
		self.forward()

	def add_placeholder(self):
		with tf.variable_scope("placeholder",reuse=tf.AUTO_REUSE):
			self.premise=tf.placeholder(shape=[None,self.seq_length_p],dtype=tf.int32)
			self.hypothesis=tf.placeholder(shape=[None,self.seq_length_h],dtype=tf.int32)
			self.premise_length=tf.placeholder(shape=[None],dtype=tf.int32)
			self.hypothesis_length=tf.placeholder(shape=[None],dtype=tf.int32)
			self.y=tf.placeholder(shape=[None],dtype=tf.int32)
			self.keep_prob=tf.placeholder(dtype=tf.float32,shape=[])
			self.p_mask=tf.cast(tf.sequence_mask(self.premise_length,maxlen=self.seq_length_p),dtype=tf.float32)
			self.h_mask=tf.cast(tf.sequence_mask(self.hypothesis_length,maxlen=self.seq_length_h),dtype=tf.float32)


	def add_embedding(self):
		self.add_placeholder()
		with tf.variable_scope("embedding",reuse=tf.AUTO_REUSE):
			# self.embedding_matrix=tf.get_variable("embeddings",shape=[self.embed_size,self.embed_dim],
			# 	initializer=tf.random_uniform_initializer(-1.0,1.0))
			self.p_embed=tf.nn.embedding_lookup(params=self.embedding_matrix,ids=self.premise)
			self.h_embed=tf.nn.embedding_lookup(params=self.embedding_matrix,ids=self.hypothesis)
			#(batch_size,sq_length,embedding_dim)
			self.p_embed=tf.expand_dims(tf.transpose(self.p_embed,perm=[0,2,1]),axis=-1)
			self.h_embed=tf.expand_dims(tf.transpose(self.h_embed,perm=[0,2,1]),axis=-1)
			#(batch_size,embed_dim,seq_length,1)

	def get_attention(self,p_tensor,h_tensor,epsilon=1e-4):
		'''
		A的第i行意味着句子premise的第i个单词对句子hypothesis注意力分布
		A的第j列意味着句子hypothesis的第j个单词对句子premise的注意力分布
		在行的方向上A可以被视为是premise的新的特征，因为Ａ的每一行是句子premise的某一个单词的新的特征向量
		在列的方向上Ａ可以被视为是hypothesis的新的特征，因为A的每一列是句子hypothesis的一个单词的新的特征向量
		'''
		#euclidean=tf.sqrt(tf.reduce_sum(tf.square(p_tensor-tf.transpose(h_tensor,perm=[0,1,3,2])),axis=1))
		#euclidean=tf.reduce_sum(tf.square(p_tensor-tf.transpose(h_tensor,perm=[0,1,3,2])),axis=1)
		euclidean=tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(p_tensor-tf.transpose(h_tensor,
			perm=[0,1,3,2])),axis=1),epsilon))
		#(batch_size,seq_length_p,seq_length_h)
		return 1/(1.0+euclidean)

	def add_abcnn2(self,p_tensor,h_tensor):
		#(batch_size,features,seq_length_P+2,1)
		#(batch_size,features,seq_length_h+2,1)
		p_tensor*=tf.expand_dims(tf.expand_dims(tf.pad(self.p_mask,[[0,0],[1,1]]),
			axis=1),axis=-1)
		h_tensor*=tf.expand_dims(tf.expand_dims(tf.pad(self.h_mask,[[0,0],[1,1]]),
			axis=1),axis=-1)

		A=self.get_attention(p_tensor,h_tensor)#(batch_size,p+2,h+2)
		p_attention=tf.reduce_sum(A,axis=2)#(batch_size,p+2)
		h_attention=tf.reduce_sum(A,axis=1)#(batch_size,h+2)
		# assert p_attention.shape[1]==p_tensor.shape[2]
		# assert h_attention.shape[1]==h_tensor.shape[2]
		p_attention=tf.expand_dims(tf.expand_dims(p_attention,axis=1),axis=-1)
		h_attention=tf.expand_dims(tf.expand_dims(h_attention,axis=1),axis=-1)
		#(batch_size,1,sep_length+2,1)
		assert p_attention.shape[2]==p_tensor.shape[2]
		assert h_attention.shape[2]==h_tensor.shape[2]
		seq_length=p_tensor.shape[2]-self.filter_width+1
		p_pools=[]
		h_pools=[]
		for i in range(seq_length):
			p_pools.append(tf.reduce_sum(p_attention[:,:,i:i+self.filter_width,:]*p_tensor[:,:,i:i+self.filter_width,:],
				axis=2,keep_dims=True))#(batch_size,features,1,1)
			h_pools.append(tf.reduce_sum(h_attention[:,:,i:i+self.filter_width,:]*h_tensor[:,:,i:i+self.filter_width,:],
				axis=2,keep_dims=True))
		p_result=tf.concat(p_pools,axis=2)#(batch_size,features,seq_length,1)
		h_result=tf.concat(h_pools,axis=2)
		return p_result,h_result

	def add_abcnn1(self,p_tensor,h_tensor):
		#(batch_size,features,seq_length,1)
		features=p_tensor.shape[1]
		assert features==h_tensor.shape[1]
		p_tensor*=tf.expand_dims(tf.expand_dims(self.p_mask,axis=1),axis=-1)
		h_tensor*=tf.expand_dims(tf.expand_dims(self.h_mask,axis=1),axis=-1)

		A=self.get_attention(p_tensor,h_tensor)#(batch_size,seq_length_p,seq_length_h)
		seq_len=A.shape[-1]
		with tf.variable_scope("abcnn1",reuse=tf.AUTO_REUSE):
			W=tf.get_variable(name="W",shape=[seq_len,features],initializer=tf.random_uniform_initializer(-1.0,1.0))
			p_attention=tf.matrix_transpose(tf.einsum("ijk,kl->ijl",A,W))#(batch_size,seq_length_p,features)
			h_attention=tf.matrix_transpose(tf.einsum("ijk,kl->ijl",tf.transpose(A,perm=[0,2,1]),W))

		p_attention=tf.expand_dims(p_attention,axis=-1)
		h_attention=tf.expand_dims(h_attention,axis=-1)
		new_p=tf.concat([p_attention,p_tensor],axis=-1)
		new_h=tf.concat([h_attention,h_tensor],axis=-1)
		return new_p,new_h

	def wide_convolution(self,input_tensor):
		#(batch_size,features,seq_length,1 or 2)
		'''
		tf.nn.conv2d(input,filter,strides,padding)
		input a 4D tensor(NHWC),filter a 4D tensor
		'''
		features=input_tensor.shape[1]
		in_channels=input_tensor.shape[-1]
		pad_tensor=tf.pad(input_tensor,paddings=[[0,0],[0,0],[self.filter_width-1,self.filter_width-1],[0,0]])
		kernel=tf.get_variable(name="kernel",shape=[features,self.filter_width,in_channels,self.conv_filters],
			initializer=tf.keras.initializers.glorot_normal(),dtype=tf.float32)
		conv_out=tf.nn.conv2d(input=pad_tensor,filter=kernel,strides=[1,1,1,1],padding="VALID")
		#(batch_size,1,seq_length+2,conv_filters)
		biases=tf.get_variable("baises",shape=[self.conv_filters],dtype=tf.float32,
			initializer=tf.keras.initializers.glorot_normal())
		out=tf.nn.tanh(conv_out+biases)
		return tf.transpose(out,perm=[0,3,2,1])

	def w_ap_layer(self,p_tensor,h_tensor):
		#(batch_size,seq_length+2,features,1)
		p_w_ap=tf.nn.avg_pool(p_tensor,ksize=[1,self.filter_width,1,1],strides=[1,1,1,1],padding="VALID")
		h_w_ap=tf.nn.avg_pool(h_tensor,ksize=[1,self.filter_width,1,1],strides=[1,1,1,1],padding="VALID")
		return p_w_ap,h_w_ap


	def forward(self):
		self.add_embedding()
		p_tensor=tf.nn.dropout(self.p_embed,keep_prob=self.keep_prob)
		h_tensor=tf.nn.dropout(self.h_embed,keep_prob=self.keep_prob)

		assert p_tensor.shape[-1]==h_tensor.shape[-1]==1
		if self.abcnn1:
			p_tensor,h_tensor=self.add_abcnn1(p_tensor,h_tensor)
			assert p_tensor.shape[-1]==h_tensor.shape[-1]==2

		with tf.variable_scope("conv",reuse=tf.AUTO_REUSE):
			p_conv=self.wide_convolution(p_tensor)
			h_conv=self.wide_convolution(h_tensor)
		assert p_conv.shape[-1]==h_conv.shape[-1]==1
		# p_conv=tf.layers.batch_normalization(p_conv)
		# h_conv=tf.layers.batch_normalization(h_conv)
		#(batch_size,features,seq_length+2,1)
		if self.abcnn2:
			p_conv,h_conv=self.add_abcnn2(p_conv,h_conv)
			#(batch_size,features,seq_length,1)
		else:
			p_conv,h_conv=self.w_ap_layer(p_conv,h_conv)

		#(batch_size,features,seq_length,1)
		assert p_conv.shape[-1]==1==h_conv.shape[-1]
		seq_length_P=p_conv.shape[2]
		seq_length_H=h_conv.shape[2]
		all_ap_p=tf.nn.avg_pool(p_conv,ksize=[1,1,seq_length_P,1],strides=[1,1,1,1],padding="VALID")
		all_ap_h=tf.nn.avg_pool(h_conv,ksize=[1,1,seq_length_H,1],strides=[1,1,1,1],padding="VALID")
		assert all_ap_h.shape[-2]==all_ap_h.shape[-1]==1
		features=all_ap_h.shape[1]
		assert features==all_ap_p.shape[1]
		all_ap_p=tf.reshape(all_ap_p,shape=[-1,features])
		all_ap_h=tf.reshape(all_ap_h,shape=[-1,features])

		#(batch_size,features)
		fc_input=tf.concat([all_ap_p,all_ap_h],axis=-1)
		fc_input=tf.nn.dropout(fc_input,keep_prob=self.keep_prob)
		self.fc_out=tf.layers.dense(fc_input,units=256,activation=tf.nn.tanh)
		self.train()
		
	def train(self):
		self.logits=tf.layers.dense(self.fc_out,units=self.num_classes)
		labels=tf.one_hot(self.y,self.num_classes)
		cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,logits=self.logits))
		weights_=[v for v in tf.trainable_variables() if ("W" in v.name) or ("kernel" in v.name)]
		l2_loss=tf.add_n([tf.nn.l2_loss(w) for w in weights_])*0.01
		self.loss=l2_loss+cross_entropy
		self.softmaxed_=tf.nn.softmax(self.logits)
		self.prediction=tf.cast(tf.argmax(self.softmaxed_,axis=-1),dtype=tf.int32)

		self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.y,self.prediction),dtype=tf.float32))

		t_vars=tf.trainable_variables()
		grads,_=tf.clip_by_global_norm(tf.gradients(self.loss,t_vars),5)
		optimizer=tf.train.AdamOptimizer(0.005)
		self.train_op=optimizer.apply_gradients(zip(grads,t_vars))












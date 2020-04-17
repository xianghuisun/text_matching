from ABCNN3 import ABCNN

class EMSI_model:
	def __init__(self,first_hidden_dim=64,second_hidden_dim=64,fc_relu_dim=128,fc_tanh_dim=64):
		self.first_hidden_dim=first_hidden_dim
		self.second_hidden_dim=second_hidden_dim
		self.fc_relu_dim=fc_relu_dim
		self.fc_tanh_dim=fc_tanh_dim

	def fc_relu_layer(self,x):
		#(batch_size,seq_a,features)
		in_dim=int(x.shape[-1])
		with tf.variable_scope("relu_fc_layer",reuse=tf.AUTO_REUSE) as scope:
			W=tf.get_variable(name="weights",shape=[in_dim,self.fc_relu_dim],
				initializer=tf.random_uniform_initializer(-1.0,1.0))
			scope.reuse_variables()
		with tf.variable_scope("relu_fc_layer",reuse=tf.AUTO_REUSE) as scope:
			b=tf.get_variable(name="biases",shape=[self.fc_relu_dim],
				initializer=tf.random_uniform_initializer(-0.1,0.1))
			scope.reuse_variables()
		return tf.nn.bias_add(tf.einsum("ijk,kl->ijl",x,W),b)


	def bilstm_layer(self,input_,input_seq_length):
		cell_fw=tf.nn.rnn_cell.LSTMCell(self.first_hidden_dim)
		cell_bw=tf.nn.rnn_cell.LSTMCell(self.first_hidden_dim)

		outputs,(state_fw,state_bw)=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,
			inputs=input_,sequence_length=input_seq_length,dtype=tf.float32)
		return tf.concat(values=[outputs[0],outputs[1]],axis=-1)

	def get_attention(self,x1,x2):
		#(batch_size,seq_1,features) (batch_size,seq_2,features)
		attention_matrix=tf.matmul(x1,x2,transpose_b=True)#(batch_size,seq_1,seq_2)
		attention_matrix_t=tf.transpose(attention_matrix,perm=[0,2,1])#(batch_size,seq_2,seq_1)
		x1_attention=tf.matmul(tf.nn.softmax(attention_matrix),x2)
		x2_attention=tf.matmul(tf.nn.softmax(attention_matrix_t),x1)
		return x1_attention,x2_attention


	def forward(self,premise,hypothesis,premise_seq_length,hypothesis_seq_length):
		#abcnn output---->(batch_size,seq_a,50) (batch_size,seq_b,50)
		with tf.variable_scope("encode_lstm_layer",reuse=tf.AUTO_REUSE):
			p=self.bilstm_layer(premise,premise_seq_length)#(batch_size,seq_a,hidden_dim*2)
		with tf.variable_scope("encode_lstm_layer",reuse=tf.AUTO_REUSE):
			h=self.bilstm_layer(hypothesis,hypothesis_seq_length)
		p_attention,h_attention=self.get_attention(p,h)
		#assert p_attention.shape[1]==p.shape[1] and p_attention.shape[2]==p.shape[2]
		m_a=tf.concat(values=[p_attention,p,tf.subtract(p,p_attention),p_attention*p,premise],axis=-1)
		#assert h_attention.shape[1]==h.shape[1] and h_attention.shape[2]==h.shape[2]
		m_b=tf.concat(values=[h_attention,h,tf.subtract(h,h_attention),h_attention*h,hypothesis],axis=-1)
		#8*first_hidden_dim+filters 8*64+50 562
		v_a=self.fc_relu_layer(m_a)
		v_b=self.fc_relu_layer(m_b)#(batch_size,seq_b,128)
		with tf.variable_scope("compose_lstm_layer",reuse=tf.AUTO_REUSE):
			compose_a=self.bilstm_layer(v_a,premise_seq_length)
		with tf.variable_scope("compose_lstm_layer",reuse=tf.AUTO_REUSE):
			compose_b=self.bilstm_layer(v_b,hypothesis_seq_length)
		#(batch_size,seq_,64*2)
		a_avg=tf.reduce_mean(compose_a,axis=1)
		a_max=tf.reduce_max(compose_a,axis=1)
		b_avg=tf.reduce_mean(compose_b,axis=1)
		b_max=tf.reduce_max(compose_b,axis=1)
		#(batch_size,128)#4*128==512
		concat_ab=tf.concat([a_avg,a_max,b_avg,b_max],axis=-1)
		output=tf.layers.dense(concat_ab,units=self.fc_tanh_dim,activation=tf.nn.tanh)
		#(batch_size,64)
		return output






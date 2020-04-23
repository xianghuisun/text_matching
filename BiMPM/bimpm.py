import tensorflow as tf

def deal_with_small_value(n,d,epsilon=1e-5):
    d=tf.cast(d>epsilon,tf.float32)*d+epsilon*tf.cast(d<=epsilon,dtype=tf.float32)
    return n/d

# def get_attention(v1,v2):
#     '''
#     v1.shape==(batch,seqlen1,dim) and v2.shape==(batch,seqlen2,dim)
#     '''
#     numerator=tf.matmul(v1,tf.transpose(v2,perm=[0,2,1]))#(batch,seqlen1,seqlen2)
#     denominator=tf.norm(v1,axis=2,keep_dims=True)*tf.transpose(tf.norm(v2,axis=2,keep_dims=True),perm=[0,2,1])
#     return deal_with_small_value(numerator,denominator)#(abtch,seqken1,seqlen2)

def get_cosine_distance(v1,v2,cosine_norm=False,epsilon=1e-5):
	#v1.shape==(batch,seqlen_1,dim),v2.shape==(batch,seqlen_2,dim)
	v1=tf.expand_dims(v1,axis=2)#(batch,seqlen_1,1,dim)
	v2=tf.expand_dims(v2,axis=1)#(batch,1,seqlen_2,dim)
	numerator=tf.reduce_sum(tf.multiply(v1,v2),axis=-1)#(batch,seqlen_1,seqlen_2)
	if not cosine_norm:
		return tf.tanh(numerator)
	v1_norm=tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v1),axis=-1),epsilon))#(batch,seqlen_1,1)
	v2_norm=tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v2),axis=-1),epsilon))#(batch,1,seqlen_2)
	return numerator/v1_norm/v2_norm#(batch_size,seqlen1,seqlen2)


def mp_matching_func(v1,v2,w,l=20,epsilon=1e-5):
    '''
    v1.shape==(batch,seqlen,dim)==v2.shape and w.shape==(l,dim)
    '''
    #seqlen=int(v1.shape[1])
    w=tf.transpose(w,perm=[1,0])#(dim,l)
    w=tf.expand_dims(tf.expand_dims(w,axis=0),axis=0)#(1,1,dim,l)
    assert w.shape[-1]==l and w.shape[-2]==v1.shape[-1]
    v1=w*tf.stack([v1]*l,axis=-1)#(batch_size,seqlen,dim,l)
    assert len(v2.shape)==3
    v2=w*tf.stack([v2]*l,axis=-1)#(batch_size,seqlen,dim,l)
    # else:
    #     assert len(v2.shape)==2#(batch_size,dim)
    #     v2=tf.stack([v2]*seqlen,axis=1)#(batch_size,seqlen,dim)
    #     v2=w*tf.stack([v2]*l,axis=-1)#(batch_size,seqlen1,dim,l)
    numerator=tf.reduce_sum(tf.multiply(v1,v2),axis=2)
    v1_norm=tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v1),axis=2),epsilon))
    v2_norm=tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v2),axis=2),epsilon))
    return numerator/v1_norm/v2_norm#(batch_size,seqlen,l)

#maxfull_match不用mp_matching_func函数


def mp_matching_func_pairwise(v1,v2,w,l=20,epsilon=1e-5):
    '''
    (batch_size,seqlen1,dim) and (batch_size,seqlen2,dim) and (l,dim)
    ''' 
    w=tf.expand_dims(tf.expand_dims(w,axis=0),axis=2)#(1,l,1,dim)
    v1=w*tf.stack([v1]*l,axis=1)
    v2=w*tf.stack([v2]*l,axis=1)
    assert w.shape[-1]==v1.shape[-1]==v2.shape[-1] and w.shape[1]==l
    v1_norm=tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v1),axis=-1,keep_dims=True),epsilon))#(batch,l,seqlen1,1)
    v2_norm=tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v2),axis=-1,keep_dims=True),epsilon))#(battch,l,seqlen2,1)
    #(batch,l,seqlen1,dim)*(abtch,l,dim,seqlen2)----->(batch,l,seqlen1,seqlen2)
    numerator=tf.matmul(v1,tf.transpose(v2,perm=[0,1,3,2]))#(batch,l,seqlen1,seqlen2)
    denominator=tf.matmul(v1_norm,tf.transpose(v2_norm,perm=[0,1,3,2]))
    result=numerator/denominator#(batch,,l,seqlen1,seqlen2)
    return tf.transpose(result,perm=[0,2,3,1])#(batch,seqlen1,selqne2,l)
#目的就是把seqlen2个l向量取最大值


class Model:
    def __init__(self,args,embedding_matrix):
        self.context_hidden_dim=args.context_hidden_dim
        self.aggregation_hidden_dim=args.aggregation_hidden_dim
        self.num_classes=args.num_classes
        self.l=args.l
        self.args=args
        self.fc_dim=args.fc_dim
        self.w1=tf.Variable(tf.random_normal(shape=[self.l,self.context_hidden_dim],dtype=tf.float32))
        self.w2=tf.Variable(tf.random_normal(shape=[self.l,self.context_hidden_dim],dtype=tf.float32))
        self.w3=tf.Variable(tf.random_normal(shape=[self.l,self.context_hidden_dim],dtype=tf.float32))
        self.w4=tf.Variable(tf.random_normal(shape=[self.l,self.context_hidden_dim],dtype=tf.float32))
        self.w5=tf.Variable(tf.random_normal(shape=[self.l,self.context_hidden_dim],dtype=tf.float32))
        self.w6=tf.Variable(tf.random_normal(shape=[self.l,self.context_hidden_dim],dtype=tf.float32))
        self.w7=tf.Variable(tf.random_normal(shape=[self.l,self.context_hidden_dim],dtype=tf.float32))
        self.w8=tf.Variable(tf.random_normal(shape=[self.l,self.context_hidden_dim],dtype=tf.float32))
        embedding_matrix=tf.Variable(embedding_matrix,dtype=tf.float32)
        random_matrix=tf.Variable(tf.random_normal(shape=[args.embed_size,args.embed_dim],dtype=tf.float32))
        self.embed_matrix=tf.concat([embedding_matrix,random_matrix],axis=-1)#(vocab_size,600)
        self.add_placeholder()
        self.forward()
    def add_placeholder(self):
        args=self.args
        self.premise = tf.placeholder(name='p', shape=(None, args.max_seq_len), dtype=tf.int32)
        self.hypothesis = tf.placeholder(name='h', shape=(None, args.max_seq_len), dtype=tf.int32)
        self.premise_length=tf.placeholder(shape=[None],dtype=tf.int32)
        self.hypothesis_length=tf.placeholder(shape=[None],dtype=tf.int32)
        self.p_mask=tf.expand_dims(tf.cast(tf.sequence_mask(self.premise_length,maxlen=args.max_seq_len),
                                           dtype=tf.float32),axis=-1)
        self.h_mask=tf.expand_dims(tf.cast(tf.sequence_mask(self.hypothesis_length,maxlen=args.max_seq_len),
                                           dtype=tf.float32),axis=-1)

        self.y = tf.placeholder(name='y', shape=(None,), dtype=tf.int32)
        self.keep_prob = tf.placeholder(name='keep_prob', dtype=tf.float32)
    def embedding_layer(self,x):
        return tf.nn.embedding_lookup(params=self.embed_matrix,ids=x)
    def dropout_op(self,x):
        return tf.nn.dropout(x,keep_prob=self.keep_prob)
    def bilstm_layer(self,x,x_length,dim):
        cell_fw=tf.nn.rnn_cell.LSTMCell(num_units=dim)
        cell_bw=tf.nn.rnn_cell.LSTMCell(num_units=dim)
        return tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs=x,sequence_length=x_length,dtype=tf.float32)
    def forward(self):
        p_embed=self.embedding_layer(self.premise)
        h_embed=self.embedding_layer(self.hypothesis)
        p_embed=self.dropout_op(p_embed)
        h_embed=self.dropout_op(h_embed)
        
        with tf.variable_scope("context_layer",reuse=tf.AUTO_REUSE):
            (p_fw,p_bw),_=self.bilstm_layer(p_embed,self.premise_length,dim=self.context_hidden_dim)
        with tf.variable_scope("context_layer",reuse=tf.AUTO_REUSE):
            (h_fw,h_bw),_=self.bilstm_layer(h_embed,self.hypothesis_length,dim=self.context_hidden_dim)
        p_fw=p_fw*self.p_mask
        p_bw=p_bw*self.p_mask
        h_fw=h_fw*self.h_mask
        h_bw=h_bw*self.h_mask
        #Full Match:
        fm_p_fw=mp_matching_func(p_fw,tf.expand_dims(h_fw[:,-1,:],axis=1),self.w1)#(batch_size,seqlen_p,l)
        fm_p_bw=mp_matching_func(p_bw,tf.expand_dims(h_bw[:,0,:],axis=1),self.w2)#(batch_size,seqlen_p,l)
        fm_h_fw=mp_matching_func(h_fw,tf.expand_dims(p_fw[:,-1,:],axis=1),self.w1)#(batch_size,seqlen_h,l)
        fm_h_bw=mp_matching_func(h_bw,tf.expand_dims(p_bw[:,0,:],axis=1),self.w2)#(batch_size,seqlen_h,l)
        
        #MaxPooling Match:
        max_fw=mp_matching_func_pairwise(p_fw,h_fw,self.w3)#(batch_size,seqlen_p,seqlen_h,l)
        max_bw=mp_matching_func_pairwise(p_bw,h_bw,self.w4)#(batch_size,seqlen_p,seqlen_h,l)
        p_max_fw=tf.reduce_max(max_fw,axis=2)#(batch_size,seqlen_p,l)
        p_max_bw=tf.reduce_max(max_bw,axis=2)#(batch_size,seqlen_p,l)
        h_max_fw=tf.reduce_max(max_fw,axis=1)#(batch_size,seqlen_h,l)
        h_max_bw=tf.reduce_max(max_bw,axis=1)#(batch_size,seqlen_h,l)
        #Attentive match
        attention_fw=get_cosine_distance(p_fw,h_fw)#(batch_size,seqlen_p,seqlen_h)
        attention_bw=get_cosine_distance(p_bw,h_bw)#(batch_size,seqlen_p,seqlen_h)
        #h_fw.shape==h_bw.shape==(batch,seqlen_h,dim)
        #p_fw.shape==p_bw.shape==(batch,seqlen_p,dim)
        attention_h_fw=tf.expand_dims(h_fw,axis=1)*tf.expand_dims(attention_fw,axis=-1)
        #(batch,1,seqlen_h,dim)*(batch,seqlen_p,seqlen_h,1)---->(batch_size,seqlen_p,seqlen_h,dim)
        attention_h_bw=tf.expand_dims(h_bw,axis=1)*tf.expand_dims(attention_bw,axis=-1)

        attention_p_fw=tf.expand_dims(p_fw,axis=2)*tf.expand_dims(attention_fw,axis=-1)
        #(batch,seqlen_p,1,dim)*(batch,seqlen_p,seqlen_h,1)
        attention_p_bw=tf.expand_dims(p_bw,axis=2)*tf.expand_dims(attention_bw,axis=-1)
        #(batch_size,seqlen_p,seqlen,h,dim)
        h_fw_mean=deal_with_small_value(tf.reduce_sum(attention_h_fw,axis=2),
        	tf.reduce_sum(attention_fw,axis=2,keep_dims=True))
        #(batch,seqlen_p,dim)/(batch_size,seqlen_p,1)----->(bach,seqlen_p,dim)
        h_bw_mean=deal_with_small_value(tf.reduce_sum(attention_h_bw,axis=2),
        	tf.reduce_sum(attention_bw,axis=2,keep_dims=True))
        p_fw_mean=deal_with_small_value(tf.reduce_sum(attention_p_fw,axis=1),
                tf.transpose(tf.reduce_sum(attention_fw,axis=1,keep_dims=True),perm=[0,2,1]))
        #(batch_size,seqlen_h,dim)/(batch_size,seqlen_h,1)*---------->(batch,seqlen_h,dim)
        p_bw_mean=deal_with_small_value(tf.reduce_sum(attention_p_bw,axis=1),
                tf.transpose(tf.reduce_sum(attention_bw,axis=1,keep_dims=True),perm=[0,2,1]))
        #p_bw_mean.shape==(batch_size,seqlen_h,dim)==p_fw_mean.shape==h_fw.shape==h_bw.shape
        #(h_fw_mean).shape==(batch_size,seqlen_p,dim)==h_bw_mean.shape==p_fw.shape==p_bw.shape
        am_p_fw=mp_matching_func(p_fw,h_fw_mean,self.w5)#(batch,seqlen_p,l)
        am_p_bw=mp_matching_func(p_bw,h_bw_mean,self.w6)#(batch,seqlen_p,l)
        am_h_fw=mp_matching_func(h_fw,p_fw_mean,self.w5)#(batch,seqlen_h,l)
        am_h_bw=mp_matching_func(h_bw,p_bw_mean,self.w6)#(batch,seqlen_h,l)
        
        #max Atentive matching
        max_h_fw_att=tf.reduce_max(attention_h_fw,axis=2)#(batch_size,seqlen_p,dim)
        max_h_bw_att=tf.reduce_max(attention_h_bw,axis=2)#(batch_size,seqlen_p,dim)
        max_p_fw_att=tf.reduce_max(attention_p_fw,axis=1)#(batch_size,seqlen_h,dim)
        max_p_bw_att=tf.reduce_max(attention_p_bw,axis=1)#(batch_size,seqlen_h,dim)

        MaxAtt_p_fw=mp_matching_func(p_fw,max_h_fw_att,self.w7)#(batch,seqlen_p,l)
        MaxAtt_p_bw=mp_matching_func(p_bw,max_h_bw_att,self.w8)#(batch,seqlen_p,l)
        MaxAtt_h_fw=mp_matching_func(h_fw,max_p_fw_att,self.w7)#(batch,seqlen_h,l)
        MaxAtt_h_bw=mp_matching_func(h_bw,max_p_bw_att,self.w8)#(batch,seqlen_h,l)

        matched_p=tf.concat([fm_p_fw,fm_p_bw,p_max_fw,p_max_bw,am_p_fw,am_p_bw,MaxAtt_p_fw,MaxAtt_p_bw],axis=-1)#(batch,seqlen_p,8*l)
        matched_h=tf.concat([fm_h_fw,fm_h_bw,h_max_fw,h_max_bw,am_h_fw,am_h_bw,MaxAtt_h_fw,MaxAtt_h_bw],axis=-1)#(batch,seqlen_h,8*l)
        matched_p=self.dropout_op(matched_p)
        matched_h=self.dropout_op(matched_h)
        
        with tf.variable_scope("aggregation_layer",reuse=tf.AUTO_REUSE):
            (p_fw,p_bw),_=self.bilstm_layer(matched_p,self.premise_length,dim=self.aggregation_hidden_dim)
        with tf.variable_scope("aggregation_layer",reuse=tf.AUTO_REUSE):
            (h_fw,h_bw),_=self.bilstm_layer(matched_h,self.hypothesis_length,dim=self.aggregation_hidden_dim)
        p_fw=p_fw*self.p_mask
        p_bw=p_bw*self.p_mask
        h_fw=h_fw*self.h_mask
        h_bw=h_bw*self.h_mask
        aggregation_out=tf.concat([p_fw[:,-1,:],p_bw[:,0,:],
                                    h_fw[:,-1,:],h_bw[:,0,:]],axis=-1)#(batch,2*aggregation_dim*4)
        aggregation_out=self.dropout_op(aggregation_out)
        fc_out=tf.layers.dense(aggregation_out,units=self.fc_dim,activation=tf.nn.tanh)
        self.logits=tf.layers.dense(fc_out,units=self.num_classes)
        self.train()
    def train(self):
        labels=tf.one_hot(self.y,self.num_classes)
        loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,labels=labels)
        self.loss=tf.reduce_mean(loss)
        optimizer=tf.train.AdamOptimizer(0.001)
        grads=tf.gradients(self.loss,tf.trainable_variables())
        grads,_=tf.clip_by_global_norm(grads,5.0)
        self.train_op=optimizer.apply_gradients(zip(grads,tf.trainable_variables()))
        self.prediction=tf.cast(tf.argmax(tf.nn.softmax(self.logits),axis=-1),dtype=tf.int32)
        self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.prediction,self.y),dtype=tf.float32))
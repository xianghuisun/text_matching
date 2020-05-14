import tensorflow as tf

def get_cosine_distance(v1,v2):
    '''
    v1.shape==(batch_size,seq_length1,dim)
    v2.shape==(batch_size,seq_length2,dim)
    return output.shape==(batch_size,seq_length1,seq_length2)
    '''
    v1=tf.expand_dims(v1,axis=2)#(batch_size,seq_length1,1,dim)
    v2=tf.expand_dims(v2,axis=1)#(batch_size,1,seq_length2,dim)
    numerator=tf.reduce_sum(tf.multiply(v1,v2),axis=-1)#(batch_size,seq_length1,seq_length2)
    v1_norm=tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v1),axis=-1),1e-5))
    v2_norm=tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v2),axis=-1),1e-5))
    return numerator/v1_norm/v2_norm#(batch_size,seq_length1,seq_length2)


def get_multi_match(v1,v2,W):
    '''
    v1.shape==(batch_size,seq_length,dim)
    v2.shape==(batch_size,seq_length or 1,dim)
    W.shape==(l,dim)
    return outputs.shape==(batch_size,seq_length,l)
    '''
    W=tf.transpose(W,perm=[1,0])#(dim,l)
    W=tf.expand_dims(tf.expand_dims(W,axis=0),axis=0)#(1,1,dim,l)
    #assert W.shape[-1]==l and W.shape[-2]==v1.shape[-1]
    v1=W*tf.expand_dims(v1,axis=-1)#(batch_size,seqlen,dim,l)
    assert len(v2.shape)==3
    v2=W*tf.expand_dims(v2,axis=-1)#(batch_size,seqlen,dim,l)

    numerator=tf.reduce_sum(tf.multiply(v1,v2),axis=2)
    v1_norm=tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v1),axis=2),1e-5))
    v2_norm=tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v2),axis=2),1e-5))
    return numerator/v1_norm/v2_norm#(batch_size,seqlen,l)

def get_multi_match_pairs(v1,v2,W):
    '''
    v1.shape==(batch_size,seq_length1,dim)
    v2.shape==(batch_size,seq_length2,dim)
    W.shape==(l,dim)
    return outputs.shape==(batch_size,l,seq_length1,seq_length2)
    '''
    v1=tf.expand_dims(v1,axis=1)
    v2=tf.expand_dims(v2,axis=1)
    W=tf.expand_dims(W,axis=0)#(1,l,dim)
    W=tf.expand_dims(W,axis=2)#(1,l,1,dim)
    v1=tf.multiply(v1,W)
    v2=tf.multiply(v2,W)
    numerator=tf.matmul(v1,tf.transpose(v2,perm=[0,1,3,2]))#(batch_size,l,seq_length1,seq_length2)
    v1_norm=tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v1),axis=-1,keep_dims=True),1e-5))
    #(batch_size,l,seq_length1,1)
    v2_norm=tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(v2),axis=-1,keep_dims=True),1e-5))
    return numerator/v1_norm/tf.transpose(v2_norm,perm=[0,1,3,2])


class Model:
    def __init__(self,args,embedding_matrix):
        self.args=args
        self.hidden_dim=args.hidden_dim
        self.num_classes=args.num_classes
        self.l=args.l
        self.rnn_layers=args.rnn_layers
        self.autoencoder_dim=args.autoencoder_dim
        self.fc_dim=args.fc_dim
        self.global_step=tf.Variable(0,trainable=False)
        self.embed_matrix=tf.Variable(name="embedding_matrix",initial_value=embedding_matrix,dtype=tf.float32)
        self.add_placeholder()
        self.forward()
        
    def add_placeholder(self):
        self.premise = tf.placeholder(name='p', shape=(None, None), dtype=tf.int32)
        self.hypothesis = tf.placeholder(name='h', shape=(None,None), dtype=tf.int32)
        self.premise_length=tf.placeholder(name="p_length",shape=[None],dtype=tf.int32)
        self.hypothesis_length=tf.placeholder(shape=[None],dtype=tf.int32)
        self.p_mask=tf.expand_dims(tf.cast(tf.sequence_mask(self.premise_length),
                                           dtype=tf.float32),axis=-1)
        self.h_mask=tf.expand_dims(tf.cast(tf.sequence_mask(self.hypothesis_length),
                                           dtype=tf.float32),axis=-1)

        self.y = tf.placeholder(name='y', shape=(None,), dtype=tf.int32)
        self.keep_prob = tf.placeholder(name='keep_prob', dtype=tf.float32)
    
    def get_perspective_W(self,name):
        with tf.variable_scope(name_or_scope="getPerspective",reuse=tf.AUTO_REUSE,initializer=tf.random_uniform_initializer(-1.0,1.0)):
            return tf.get_variable(name=name,shape=[self.l,self.hidden_dim],dtype=tf.float32)
    def dropout_layer(self,x):
        return tf.nn.dropout(x,keep_prob=self.keep_prob)
    def bilstm_layer(self,x,x_length):
        cell_fw=tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_dim)
        cell_bw=tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_dim)
        outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,cell_bw=cell_bw,inputs=x,
                                       sequence_length=x_length,dtype=tf.float32)
        return outputs
    
    def multi_perspective_layer(self,p_fw,p_bw,h_fw,h_bw):
        '''
        p_fw.shape==p_bw.shape==(batch,seq_length_p,dim)
        h in the same way
        return two tensor,each tensor has the shape of (batch_size,seq_length,self.l*8)
        '''
        W1=self.get_perspective_W(name="w1")
        W2=self.get_perspective_W(name="w2")
        full_match_p_fw=get_multi_match(p_fw,tf.expand_dims(h_fw[:,-1,:],axis=1),W=W1)
        full_match_p_bw=get_multi_match(p_bw,tf.expand_dims(h_bw[:,0,:],axis=1),W=W2)
        full_match_h_fw=get_multi_match(h_fw,tf.expand_dims(p_fw[:,-1,:],axis=1),W=W1)
        full_match_h_bw=get_multi_match(h_bw,tf.expand_dims(p_bw[:,0,:],axis=1),W=W2)
        #(batch_size,seq_length,l)
        assert full_match_p_fw.shape[-1]==self.l and full_match_h_bw.shape[-1]==self.l
        
        #MaxFull Match
        W3=self.get_perspective_W(name="w3")
        W4=self.get_perspective_W(name="w4")
        max_full_match_p_fw=get_multi_match_pairs(p_fw,h_fw,W=W3)
        max_full_match_p_bw=get_multi_match_pairs(p_bw,h_bw,W=W4)
        #(batch_size,l,seq_length1,seq_length2)
        max_full_match_h_fw=get_multi_match_pairs(h_fw,p_fw,W=W3)
        max_full_match_h_bw=get_multi_match_pairs(h_bw,p_bw,W=W4)
        #(batch_size,l,seq_length2,seq_length1)
        max_full_match_p_fw=tf.transpose(tf.reduce_max(max_full_match_p_fw,axis=-1),perm=[0,2,1])
        max_full_match_p_bw=tf.transpose(tf.reduce_max(max_full_match_p_bw,axis=-1),perm=[0,2,1])
        max_full_match_h_fw=tf.transpose(tf.reduce_max(max_full_match_h_fw,axis=-1),perm=[0,2,1])
        max_full_match_h_bw=tf.transpose(tf.reduce_max(max_full_match_h_bw,axis=-1),perm=[0,2,1])
        #batch_size,seq_length,l
        
        #Attentive Match
        p_fw_cosine=get_cosine_distance(p_fw,h_fw)
        p_bw_cosine=get_cosine_distance(p_bw,h_bw)
        p_fw_attention=tf.expand_dims(p_fw_cosine,axis=-1)*tf.expand_dims(h_fw,axis=1)
        p_bw_attention=tf.expand_dims(p_bw_cosine,axis=-1)*tf.expand_dims(h_bw,axis=1)
        #(batch,seq1,seq2,dim)
        h_fw_cosine=get_cosine_distance(h_fw,p_fw)
        h_bw_cosine=get_cosine_distance(h_bw,p_bw)
        h_fw_attention=tf.expand_dims(h_fw_cosine,axis=-1)*tf.expand_dims(p_fw,axis=1)
        h_bw_attention=tf.expand_dims(h_bw_cosine,axis=-1)*tf.expand_dims(p_bw,axis=1)
        #(batch,seq_length_h,seq_length_p,dim)
        p_fw_mean=tf.reduce_sum(p_fw_attention,axis=2)/tf.reduce_sum(p_fw_cosine,axis=-1,keep_dims=True)
        p_bw_mean=tf.reduce_sum(p_bw_attention,axis=2)/tf.reduce_sum(p_bw_cosine,axis=-1,keep_dims=True)
        h_fw_mean=tf.reduce_sum(h_fw_attention,axis=2)/tf.reduce_sum(h_fw_cosine,axis=-1,keep_dims=True)
        h_bw_mean=tf.reduce_sum(h_bw_attention,axis=2)/tf.reduce_sum(h_bw_cosine,axis=-1,keep_dims=True)
        p_fw_max=tf.reduce_max(p_fw_attention,axis=2)
        p_bw_max=tf.reduce_max(p_bw_attention,axis=2)
        h_fw_max=tf.reduce_max(h_fw_attention,axis=2)
        h_bw_max=tf.reduce_max(h_bw_attention,axis=2)
        
        W5=self.get_perspective_W(name="w5")
        W6=self.get_perspective_W(name="w6")
        #p_fw_mean.shape==p_fw.shape==p_fw_max.shape==(batch_size,seq_length_p,dim)
        attentive_match_p_fw=get_multi_match(p_fw,p_fw_mean,W=W5)
        attentive_match_p_bw=get_multi_match(p_bw,p_bw_mean,W=W6)
        attentive_match_h_fw=get_multi_match(h_fw,h_fw_mean,W=W5)
        attentive_match_h_bw=get_multi_match(h_bw,h_bw_mean,W=W6)
        
        W7=self.get_perspective_W(name="w7")
        W8=self.get_perspective_W(name="w8")
        max_attentive_p_fw=get_multi_match(p_fw,p_fw_max,W7)
        max_attentive_p_bw=get_multi_match(p_bw,p_bw_max,W8)
        max_attentive_h_fw=get_multi_match(h_fw,h_fw_max,W7)
        max_attentive_h_bw=get_multi_match(h_bw,h_bw_max,W8)
        
        p_concat=tf.concat(values=[full_match_p_fw,full_match_p_bw,max_full_match_p_fw,
                                  max_full_match_p_bw,attentive_match_p_fw,attentive_match_p_bw,
                                  max_attentive_p_fw,max_attentive_p_bw],axis=-1)
        h_concat=tf.concat(values=[full_match_h_fw,full_match_h_bw,max_full_match_h_fw,
                                  max_full_match_h_bw,attentive_match_h_fw,attentive_match_h_bw,
                                  max_attentive_h_fw,max_attentive_h_bw],axis=-1)
        return p_concat,h_concat
    
    def autoencoder_layer(self,x):
        input_dim=int(x.shape[-1])
        layer_out=tf.layers.dense(inputs=x,units=self.autoencoder_dim,activation=tf.nn.relu)
        output=tf.layers.dense(inputs=layer_out,units=input_dim,activation=tf.nn.relu)
        return layer_out,tf.reduce_mean(tf.square(output-x))/2.0
    
    def forward(self):
        p_embed=tf.nn.embedding_lookup(params=self.embed_matrix,ids=self.premise)
        h_embed=tf.nn.embedding_lookup(params=self.embed_matrix,ids=self.hypothesis)
        p=self.dropout_layer(p_embed)
        h=self.dropout_layer(h_embed)
        self.autoencoder_loss=[]
        for i in range(self.rnn_layers):
            p_old=p
            h_old=h
            with tf.variable_scope(name_or_scope=f"lstm_layer_{i}",reuse=tf.AUTO_REUSE):
                (p_fw,p_bw)=self.bilstm_layer(p_old,self.premise_length)
            with tf.variable_scope(name_or_scope=f"lstm_layer_{i}",reuse=tf.AUTO_REUSE):
                (h_fw,h_bw)=self.bilstm_layer(h_old,self.hypothesis_length)
            p_hidden=tf.concat(values=[p_fw,p_bw],axis=-1)
            h_hidden=tf.concat(values=[h_fw,h_bw],axis=-1)
            p_attention=tf.matmul(tf.nn.softmax(get_cosine_distance(p_hidden,h_hidden)),h_hidden)
            h_attention=tf.matmul(tf.nn.softmax(get_cosine_distance(h_hidden,p_hidden)),p_hidden)
            #p_attention.shape==p_hidden.shape h in the same way
            p_match,h_match=self.multi_perspective_layer(p_fw,p_bw,h_fw,h_bw)
            #p_match.shape==(batch_size,seq_length_p,8*self.l)
            p=tf.concat(values=[p_old,p_hidden,p_attention,p_match,p_hidden-p_attention,
                              p_hidden*p_attention],axis=-1)
            h=tf.concat(values=[h_old,h_hidden,h_attention,h_match,h_hidden-h_attention,
                               h_hidden*h_attention],axis=-1)
            p,autoencoder_loss_p=self.autoencoder_layer(x=p)
            h,autoencoder_loss_h=self.autoencoder_layer(x=h)
            self.autoencoder_loss.append(autoencoder_loss_p)
            self.autoencoder_loss.append(autoencoder_loss_h)
        
        p=tf.reduce_max(p,axis=1)#500
        h=tf.reduce_max(h,axis=1)#500
        interaction=tf.concat(values=[p,h,p+h,p-h,tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(p-h),axis=-1,keep_dims=True),1e-4))],axis=-1)#2001
        print(interaction.shape)
        fc_out=tf.layers.dense(inputs=interaction,units=self.fc_dim,activation=tf.nn.relu)
        self.logits=tf.layers.dense(inputs=fc_out,units=self.num_classes)
        self.train()
    def train(self):
        labels=tf.one_hot(self.y,self.num_classes)
        cross_entropy_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                                  labels=labels))
        trainable_vars=tf.trainable_variables()
        l2_regular=[]
        for v in trainable_vars:
            if "embedding_matrix" not in v.name:
                l2_regular.append(self.args.theta*tf.nn.l2_loss(v))
        self.loss=cross_entropy_loss+tf.reduce_sum(self.autoencoder_loss)+tf.reduce_sum(l2_regular)
        learning_rate=tf.train.exponential_decay(learning_rate=0.05,global_step=self.global_step,decay_steps=1000,decay_rate=0.9)
        optimizer=tf.train.AdamOptimizer(learning_rate)
        grads=tf.gradients(self.loss,trainable_vars)
        grads,_=tf.clip_by_global_norm(grads,3.0)
        self.train_op=optimizer.apply_gradients(zip(grads,trainable_vars),global_step=self.global_step)
        self.prediction=tf.cast(tf.argmax(tf.nn.softmax(self.logits),axis=-1),dtype=tf.int32)
        self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.prediction,self.y),dtype=tf.float32))
        

import tensorflow as tf
import numpy as np

def get_attention(x1,x2):
    '''
    x1.shape==(batch_size,seq_length1,dim or dim*2) x2.shape==(batch_size,seq_length2,dim or dim*2)
    return batch_size,seqLengh1,seqLnegth2
    '''
    assert x1.shape[-1]==x2.shape[-1]
    attention=tf.matmul(x1,tf.transpose(x2,perm=[0,2,1]))
    return attention

def layer_norm(inputs,epsilon=1e-6):
    with tf.variable_scope("ln",reuse=tf.AUTO_REUSE):
        features=int(inputs.shape[-1])
        mean,variance=tf.nn.moments(inputs,[-1],keep_dims=True)
        beta=tf.get_variable(name="mean",shape=[features],initializer=tf.zeros_initializer())
        gamma=tf.get_variable(name="gamma",shape=[features],initializer=tf.ones_initializer())
        normalized=(inputs-mean)/((variance+epsilon)**(0.5))
        return gamma*normalized+beta

class Model:
    def __init__(self,args,embedding_matrix):
        self.args=args
        self.num_classes=args.num_classes
        self.embed_dim=args.embed_dim
        self.embed_size=args.embed_size
        self.context_hidden_dim=args.context_hidden_dim
        self.compose_hidden_dim=args.compose_hidden_dim
        self.embed_matrix=tf.Variable(embedding_matrix,dtype=tf.float32)
        self.add_placeholder()    
        self.forward()
        
    def add_placeholder(self):
        self.premise=tf.placeholder(shape=[None,None],dtype=tf.int32)
        self.hypothesis=tf.placeholder(shape=[None,None],dtype=tf.int32)
        self.premise_length=tf.placeholder(shape=[None],dtype=tf.int32)
        self.hypothesis_length=tf.placeholder(shape=[None],dtype=tf.int32)
        self.y=tf.placeholder(shape=[None],dtype=tf.int32)
        self.p_mask=tf.expand_dims(tf.cast(tf.sequence_mask(self.premise_length),dtype=tf.float32),axis=-1)
        self.h_mask=tf.expand_dims(tf.cast(tf.sequence_mask(self.hypothesis_length),dtype=tf.float32),axis=-1)
        self.keep_prob=tf.placeholder(dtype=tf.float32,shape=[])
        #(batch_size,seqLen,1)

    def embedding_layer(self):
        self.p_embed=tf.nn.embedding_lookup(params=self.embed_matrix,ids=self.premise)
        self.h_embed=tf.nn.embedding_lookup(params=self.embed_matrix,ids=self.hypothesis)
        self.p_embed=self.dropout_layer(self.p_embed)
        self.h_embed=self.dropout_layer(self.h_embed)

    def context_bilstm_layer(self,x,x_length=None):
        cell_fw=tf.nn.rnn_cell.LSTMCell(num_units=self.context_hidden_dim)
        cell_bw=tf.nn.rnn_cell.LSTMCell(num_units=self.context_hidden_dim)
        outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs=x,sequence_length=x_length,dtype=tf.float32)
        return outputs

    def compose_bilstm_layer(self,x,x_length=None):
        cell_fw=tf.nn.rnn_cell.LSTMCell(num_units=self.compose_hidden_dim)
        cell_bw=tf.nn.rnn_cell.LSTMCell(num_units=self.compose_hidden_dim)
        outputs,states=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs=x,sequence_length=x_length,dtype=tf.float32)
        return outputs
    def dropout_layer(self,x):
        return tf.nn.dropout(x,keep_prob=self.keep_prob)
    def get_attention(self,x1,x2):
        '''
        x1.shape==(batch_size,seq_length1,dim or dim*2) x2.shape==(batch_size,seq_length2,dim or dim*2)
        return batch_size,seqLengh1,seqLnegth2
        '''
        assert x1.shape[-1]==x2.shape[-1]
        attention=tf.matmul(x1,tf.transpose(x2,perm=[0,2,1]))
        return attention

    def forward(self):
        self.embedding_layer()
        with tf.variable_scope("context",reuse=tf.AUTO_REUSE):
            (p_fw,p_bw)=self.context_bilstm_layer(self.p_embed,self.premise_length)
        with tf.variable_scope("context",reuse=tf.AUTO_REUSE):
            (h_fw,h_bw)=self.context_bilstm_layer(self.h_embed,self.hypothesis_length)
        p=tf.concat([p_fw,p_bw],axis=-1)#(batch_size,seq_length_p,context_dim*2)
        h=tf.concat([h_fw,h_bw],axis=-1)#(batch_size,seq_length_h,context_dim*2)
        p=tf.multiply(p,self.p_mask)
        h=tf.multiply(h,self.h_mask)
        p_attention=tf.matmul(tf.nn.softmax(get_attention(p,h)),h)#(batch,seqlen1,seqlen2)*(batch,seqlen2,dim)---->(batch,seqlen1,dim)
        h_attention=tf.matmul(tf.nn.softmax(get_attention(h,p)),p)#(batch,seqlen2,seqlen1)*(batch,seqlen1,dim)---->(batch,seqlen2,dim)

        m_a=tf.concat([p,p_attention,p-p_attention,tf.multiply(p,p_attention)],axis=-1)
        m_b=tf.concat([h,h_attention,h-h_attention,tf.multiply(h,h_attention)],axis=-1)
        #(batch_size,seq_len,4*conetxt_hidden_dim*2)
        #m_a=layer_norm(m_a)
        #m_b=layer_norm(m_b)
        with tf.variable_scope("compose",reuse=tf.AUTO_REUSE):
            (a_fw,a_bw)=self.compose_bilstm_layer(m_a,self.premise_length)
        with tf.variable_scope("compose",reuse=tf.AUTO_REUSE):
            (b_fw,b_bw)=self.compose_bilstm_layer(m_b,self.hypothesis_length)
        #(batch_size,seq_len,2*compose_dim)
        compose_a=tf.concat([a_fw,a_bw],axis=-1)
        compose_b=tf.concat([b_fw,b_bw],axis=-1)
        compose_a=tf.multiply(compose_a,self.p_mask)
        compose_b=tf.multiply(compose_b,self.h_mask)
        
        a_avg=tf.reduce_mean(compose_a,axis=1)
        a_max=tf.reduce_max(compose_a,axis=1)
        a_concat=tf.concat([a_avg,a_max],axis=-1)#(batch_size,4*compose_dim)

        b_avg=tf.reduce_mean(compose_b,axis=1)
        b_max=tf.reduce_max(compose_b,axis=1)
        b_concat=tf.concat([b_avg,b_max],axis=-1)#(batch_size,4*compose_dim)

        v=tf.concat([a_concat,b_concat],axis=-1)#(batch_size,8*compose_dim)
        v=self.dropout_layer(v)
        outouts=tf.layers.dense(v,units=self.args.fc_dim,activation=tf.nn.tanh)
        self.logits=tf.layers.dense(outouts,units=self.num_classes)
        self.train()

    def train(self):
        labels=tf.one_hot(self.y,self.num_classes)
        loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,labels=labels)
        self.loss=tf.reduce_mean(loss)
        optimizer=tf.train.AdamOptimizer(0.001)
        grads=tf.gradients(self.loss,tf.trainable_variables())
        grads,_=tf.clip_by_global_norm(grads,3.0)
        self.train_op=optimizer.apply_gradients(zip(grads,tf.trainable_variables()))
        self.prediction=tf.cast(tf.argmax(tf.nn.softmax(self.logits),axis=-1),dtype=tf.int32)
        self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.prediction,self.y),dtype=tf.float32))






        















import tensorflow as tf
import numpy as np


def assert_shape(x1,x2):
    assert x1.shape[1]==x2.shape[1] and x1.shape[2]==x2.shape[2]

def get_attention(x1,x2):
    '''
    x1.shape==(batch_size,seq_length1,dim or dim*2) x2.shape==(batch_size,seq_length2,dim or dim*2)
    return batch_size,seqLengh1,seqLnegth2
    '''
    assert x1.shape[-1]==x2.shape[-1]
    attention=tf.matmul(x1,tf.transpose(x2,perm=[0,2,1]))
    return attention
def deal_with_small_value(numerator,denominator,epsilon=1e-5):
    denominator=denominator*tf.cast(denominator>epsilon,dtype=tf.float32)+epsilon*tf.cast(denominator<=epsilon,dtype=tf.float32)
    return numerator/denominator

def get_cosine_distance(x1,x2):
    numerator=tf.matmul(x1,tf.transpose(x2,perm=[0,2,1]))#(batch,seqlen1,selen2)
    x1_norm=tf.sqrt(tf.reduce_sum(tf.square(x1),axis=-1,keep_dims=True))
    x2_norm=tf.sqrt(tf.reduce_sum(tf.square(x2),axis=-1,keep_dims=True))
    denominator=tf.matmul(x1_norm,tf.transpose(x2_norm,perm=[0,2,1]))
    #assert_shape(numerator,denominator)
    return deal_with_small_value(numerator,denominator)

def layer_norm(inputs,epsilon=1e-6):
    with tf.variable_scope("ln",reuse=tf.AUTO_REUSE):
        features=int(inputs.shape[-1])
        mean,variance=tf.nn.moments(inputs,[-1],keep_dims=True)
        beta=tf.get_variable(name="mean",shape=[features],initializer=tf.zeros_initializer())
        gamma=tf.get_variable(name="gamma",shape=[features],initializer=tf.ones_initializer())
        normalized=(inputs-mean)/((variance+epsilon)**(0.5))
        return gamma*normalized+beta

class Model:
    def __init__(self,args,embedding_matrix=None):
        self.args=args
        self.num_classes=args.num_classes
        self.embed_dim=args.embed_dim
        self.embed_size=args.embed_size
        self.context_hidden_dim=args.context_hidden_dim
        self.compose_hidden_dim=args.compose_hidden_dim
        if embedding_matrix==None:
            with tf.variable_scope("embed_matrix",reuse=tf.AUTO_REUSE):
                self.embed_matrix=tf.get_variable(name="embedding_matrix",initializer=tf.random_uniform_initializer(-1.0,1.0),
                        shape=[self.embed_size,self.embed_dim],dtype=tf.float32)
        else:
            self.embed_matrix=embed_matrix
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
    def dropout_layer(self,x,keep_prob=0.5):
        return tf.nn.dropout(x,keep_prob=keep_prob)
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
        p=tf.concat([p_fw,p_bw],axis=-1)
        h=tf.concat([h_fw,h_bw],axis=-1)
        p=tf.multiply(p,self.p_mask)
        h=tf.multiply(h,self.h_mask)
        p_attention=tf.matmul(tf.nn.softmax(get_attention(p,h)),h)#(batch,seqlen1,seqlen2)*(batch,seqlen2,dim)---->(batch,seqlen1,dim)
        h_attention=tf.matmul(tf.nn.softmax(get_attention(h,p)),p)#(batch,seqlen2,seqlen1)*(batch,seqlen1,dim)---->(batch,seqlen2,dim)
        assert_shape(p_attention,p)
        assert_shape(h_attention,h)
        cosine_distance=get_cosine_distance(p,h)#(batch,seq_p,seq_h)
        seq_p=int(p.shape[1])
        seq_h=int(h.shape[1])
        assert p.shape[-1]==h.shape[-1]==self.context_hidden_dim*2
        #没法用这个当你的seq_length是None的时候，
        W_p=tf.Variable(tf.random_normal(shape=[seq_h,self.context_hidden_dim*2],dtype=tf.float32))
        p_cosine=tf.einsum("ijk,kl->ijl",cosine_distance,W_p)
        assert_shape(p_cosine,p)
        W_h=tf.Variable(tf.random_normal(shape=[seq_p,self.context_hidden_dim*2],dtype=tf.float32))
        h_cosine=tf.einsum("ijk,kl->ijl",tf.transpose(cosine_distance,perm=[0,2,1]),W_h)
        assert_shape(h_cosine,h)
        
        m_a=tf.concat([p,p_attention,p-p_attention,tf.multiply(p,p_attention),p_cosine],axis=-1)
        m_b=tf.concat([h,h_attention,h-h_attention,tf.multiply(h,h_attention),h_cosine],axis=-1)
        #(batch_size,seq_len,10*conetxt_hidden_dim)
        m_a=layer_norm(m_a)
        m_b=layer_norm(m_b)
        with tf.variable_scope("compose",reuse=tf.AUTO_REUSE):
            (a_fw,a_bw)=self.compose_bilstm_layer(m_a,self.premise_length)
        with tf.variable_scope("compose",reuse=tf.AUTO_REUSE):
            (b_fw,b_bw)=self.compose_bilstm_layer(m_b,self.hypothesis_length)
        #(batch_size,seq_len,2*compose_dim)
        compose_a=tf.concat([a_fw,a_bw],axis=-1)
        compose_b=tf.concat([b_fw,b_bw],axis=-1)
        compose_a=tf.multiply(compose_a,self.p_mask)
        compose_b=tf.multiply(compose_b,self.h_mask)
        compose_a_attention=tf.matmul(tf.nn.softmax(get_attention(compose_a,compose_b)),compose_b)
        compose_b_attention=tf.matmul(tf.nn.softmax(get_attention(compose_b,compose_a)),compose_a)
        assert_shape(compose_a,compose_a_attention)
        assert_shape(compose_b,compose_b_attention)
        
        a_avg1=tf.reduce_mean(compose_a,axis=1)
        a_avg2=tf.reduce_mean(compose_a_attention,axis=1)
        a_max1=tf.reduce_max(compose_a,axis=1)
        a_max2=tf.reduce_max(compose_a_attention,axis=1)
        a_concat=tf.concat([a_avg1,a_avg2,a_max2,a_max1],axis=-1)#(batch_size,8*compose_dim)

        b_avg1=tf.reduce_mean(compose_b,axis=1)
        b_avg2=tf.reduce_mean(compose_b_attention,axis=1)
        b_max1=tf.reduce_max(compose_b,axis=1)
        b_max2=tf.reduce_max(compose_b_attention,axis=1)
        b_concat=tf.concat([b_avg2,b_avg1,b_max2,b_max1],axis=-1)#(batch_size,8*compose_dim)

        v=tf.concat([a_concat,b_concat],axis=-1)#(batch_size,16*compose_dim)
        v=self.dropout_layer(v)
        outouts=tf.layers.dense(v,units=512,activation=tf.nn.tanh)
        self.logits=tf.layers.dense(outouts,units=self.num_classes)
        self.train()

    def train(self):
        labels=tf.one_hot(self.y,self.num_classes)
        loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,labels=labels)
        self.loss=tf.reduce_mean(loss)
        optimizer=tf.train.AdamOptimizer(0.001)
        grads=tf.gradients(self.loss,tf.trainable_variables())
        grads,_=tf.clip_by_global_norm(grads,2.0)
        self.train_op=optimizer.apply_gradients(zip(grads,tf.trainable_variables()))
        self.prediction=tf.cast(tf.argmax(tf.nn.softmax(self.logits),axis=-1),dtype=tf.int32)
        self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.prediction,self.y),dtype=tf.float32))






        















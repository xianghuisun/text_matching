import tensorflow as tf

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

class Model:
    def __init__(self,args,embedding_matrix):
        self.hidden_dim=args.hidden_dim#100
        self.num_classes=args.num_classes#2
        self.args=args
        self.fc_dim=args.fc_dim#1000
        embedding_matrix=tf.Variable(embedding_matrix,dtype=tf.float32)
        random_matrix=tf.Variable(tf.random_normal(shape=[args.embed_size,args.embed_dim],dtype=tf.float32))
        self.embed_matrix=tf.concat([embedding_matrix,random_matrix],axis=-1)#(vocab_size,600)
        self.add_placeholder()
        self.forward()
    def add_placeholder(self):
        args=self.args
        self.premise = tf.placeholder(name='p', shape=(None, None), dtype=tf.int32)
        self.hypothesis = tf.placeholder(name='h', shape=(None,None), dtype=tf.int32)
        self.premise_length=tf.placeholder(shape=[None],dtype=tf.int32)
        self.hypothesis_length=tf.placeholder(shape=[None],dtype=tf.int32)
        self.p_mask=tf.expand_dims(tf.cast(tf.sequence_mask(self.premise_length),
                                           dtype=tf.float32),axis=-1)
        self.h_mask=tf.expand_dims(tf.cast(tf.sequence_mask(self.hypothesis_length),
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
    def autoencoder_layer(self,x):
        out_dim=int(x.shape[-1])
        layer1_out=tf.layers.dense(x,units=200,activation=tf.nn.relu)
        output=tf.layers.dense(layer1_out,units=out_dim,activation=tf.nn.tanh)
        return layer1_out,tf.reduce_mean(tf.square(output-x))
    
    def forward(self):
        p_embed=self.embedding_layer(self.premise)
        h_embed=self.embedding_layer(self.hypothesis)
        p_new=self.dropout_op(p_embed)
        h_new=self.dropout_op(h_embed)
        self.autoencoder_loss=0.0
        for i in range(5):
            p_old=p_new
            h_old=h_new
            with tf.variable_scope(f"lstm_layer_{i}",reuse=tf.AUTO_REUSE):
                (p_fw,p_bw),_=self.bilstm_layer(p_old,self.premise_length,dim=self.hidden_dim)
            with tf.variable_scope(f"lstm_layer_{i}",reuse=tf.AUTO_REUSE):
                (h_fw,h_bw),_=self.bilstm_layer(h_old,self.hypothesis_length,dim=self.hidden_dim)
            p=tf.concat([p_fw,p_bw],axis=-1)
            h=tf.concat([h_fw,h_bw],axis=-1)
            cosine_distance=get_cosine_distance(p,h)
            p_attention=tf.matmul(tf.nn.softmax(cosine_distance),h)
            h_attention=tf.matmul(tf.nn.softmax(tf.transpose(cosine_distance,perm=[0,2,1])),p)
            p_new=tf.concat(values=[p_old,p,p_attention],axis=-1)
            h_new=tf.concat(values=[h_old,h,h_attention],axis=-1)
            if i==2 or i==4:
                p_new,auto_loss_p=self.autoencoder_layer(p_new)
                h_new,auto_loss_h=self.autoencoder_layer(h_new)
                self.autoencoder_loss=self.autoencoder_loss+auto_loss_p+auto_loss_h
                
        p_pooling=tf.reduce_max(p_new,axis=1)
        h_pooling=tf.reduce_max(h_new,axis=1)
        interaction=tf.concat(values=[p_pooling,h_pooling,p_pooling+h_pooling,
                                     p_pooling-h_pooling,tf.norm(p_pooling-h_pooling,axis=-1,keep_dims=True)],
                             axis=-1)
        interaction=self.dropout_op(interaction)
        fc_out=tf.layers.dense(interaction,units=self.fc_dim,activation=tf.nn.tanh)
        self.logits=tf.layers.dense(fc_out,units=self.num_classes)
        self.train()
    def train(self):
        labels=tf.one_hot(self.y,self.num_classes)
        loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,labels=labels)
        self.loss=tf.reduce_mean(loss)+self.autoencoder_loss
        optimizer=tf.train.AdamOptimizer(0.001)
        grads=tf.gradients(self.loss,tf.trainable_variables())
        grads,_=tf.clip_by_global_norm(grads,5.0)
        self.train_op=optimizer.apply_gradients(zip(grads,tf.trainable_variables()))
        self.prediction=tf.cast(tf.argmax(tf.nn.softmax(self.logits),axis=-1),dtype=tf.int32)
        self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.prediction,self.y),dtype=tf.float32))
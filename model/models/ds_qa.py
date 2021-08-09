#coding=utf-8
'''
Created on Jan 12, 2019

@author: lyk
'''
import tensorflow as tf
import numpy as np
import time
from model.basemodel.rc_model import RCModel as basemodel

class DSQA(basemodel):
    def __init__(self,vocab,args):
        super(DSQA,self).__init__(vocab,args)
    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()#using bi-lstm passage question encoded
        self.question_att()
        self.paragraph_prob()
        self.paragraph_reader()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num)) 
    def bi_linear_att(self,passage,question,name="weight"):
        p_dim = passage.get_shape().as_list()[-1]
        q_dim = question.get_shape().as_list()[-1]
        flat_p = tf.reshape(passage, [-1, p_dim])
        weight = tf.get_variable(name, [p_dim,q_dim])
        shape=tf.shape(passage)
        out_shape = [shape[idx] for idx in range(len(passage.get_shape().as_list()) - 1)] + [q_dim]#bpd
        result_input = tf.reshape(tf.matmul(flat_p, weight), out_shape)
        outputs=tf.einsum("bpd,bd->bp",result_input,question)#bd
        return outputs
        
    def question_att(self):
        dim=self.sep_q_encodes.get_shape().as_list()[-1]
        weight=tf.get_variable(name="w1", shape=[dim])
        att=tf.nn.softmax(tf.einsum("bqd,t->bq",self.sep_q_encodes,weight),axis=-1)
        att_vector=tf.einsum("bqd,bq->bqd",self.sep_q_encodes,att)
        self.att_q=tf.reshape(tf.reduce_sum(att_vector,axis=1),shape=[-1,2*self.hidden_size])#bq
    def paragraph_prob(self):
        batch_size = tf.shape(self.start_label)[0]
        para_score=self.bi_linear_att(self.sep_p_encodes, self.att_q, name="para_weight")#bp
        self.para_logits=tf.nn.softmax(tf.reshape(tf.reduce_max(para_score,axis=-1),shape=[batch_size,-1]),axis=-1)#bn
    def paragraph_reader(self):
        batch_size = tf.shape(self.start_label)[0]
        seq_len=tf.shape(self.sep_p_encodes)[1]
        seq_para=tf.reshape(self.sep_p_encodes,shape=[batch_size,-1,seq_len,2*self.hidden_size])#bnpd
        re_para_rep=tf.einsum("bnpd,bn->bnpd",seq_para,self.para_logits)
        con_para=tf.reshape(re_para_rep,shape=[batch_size,-1,2*self.hidden_size])
        no_dup_question_encodes = tf.reshape(self.att_q,[batch_size, -1,2*self.hidden_size])[0:,0, 0:]
        self.start_probs=tf.nn.softmax(self.bi_linear_att(con_para,no_dup_question_encodes, name="answers"), axis=-1)
        self.end_probs=tf.nn.softmax(self.bi_linear_att(con_para,no_dup_question_encodes, name="answere"),axis=-1)#bp
        
        
        
        
        
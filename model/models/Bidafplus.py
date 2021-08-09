#coding=utf-8
'''
Created on Jan 12, 2019

@author: lyk
'''
import tensorflow as tf
from model.basemodel.rc_model import RCModel as basemodel
from model.layers.match_layer import SelfAttLayer
from model.layers.basic_rnn import rnn
class BIDAFPLUS(basemodel):
    def __init__(self,vocab,args):
        super(BIDAFPLUS,self).__init__(vocab,args)
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
        
    def _selfatt(self):
        self.question_att()
        self.paragraph_prob()
        attention_layer = SelfAttLayer(self.hidden_size)
        with tf.variable_scope("para_attention"):
            self_att = attention_layer.bi_linear_att(self.gated_p_encodes, self.gated_p_encodes)
            self.para_p_encodes, _ = rnn('bi-gru', self_att, self.p_length, self.hidden_size)
        with tf.variable_scope("document_attention"):
            batch_size = tf.shape(self.start_label)[0]
            doc_encodes = tf.reshape(self.para_p_encodes, [batch_size, -1, 2 * self.hidden_size])
            doc_att = attention_layer.bi_linear_att(doc_encodes, doc_encodes)
            unstack_doc=tf.reshape(doc_att,shape=[batch_size,-1,tf.shape(doc_att)[1],4*self.hidden_size])
            re_para_rep=tf.einsum("bnpd,bn->bnpd",unstack_doc,self.para_logits)
            con_para=tf.reshape(re_para_rep,shape=[batch_size,-1,4*self.hidden_size])
            no_dup_question_encodes = tf.reshape(self.att_q,[batch_size, -1,2*self.hidden_size])[0:,0, 0:]
            self.start_probs=tf.nn.softmax(self.bi_linear_att(con_para,no_dup_question_encodes, name="answers"), axis=-1)
            self.end_probs=tf.nn.softmax(self.bi_linear_att(con_para,no_dup_question_encodes, name="answere"),axis=-1)#bp
    def _decode(self):
        pass
            
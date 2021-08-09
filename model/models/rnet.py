#coding=utf-8
'''
Created on Jan 9, 2019

@author: lyk
'''
import tensorflow as tf
from model.layers.basic_rnn import rnn
from model.basemodel.rc_model import RCModel as basemodel
class RNET(basemodel):
    def __init__(self,vocab,args):
        super(RNET,self).__init__(vocab,args)
    def dense(self,inputs, hidden, use_bias=True, scope="dense"):
        with tf.variable_scope(scope):
            shape = tf.shape(inputs)
            dim = inputs.get_shape().as_list()[-1]
            out_shape = [shape[idx] for idx in range(
                len(inputs.get_shape().as_list()) - 1)] + [hidden]
            flat_inputs = tf.reshape(inputs, [-1, dim])
            W = tf.get_variable("W", [dim, hidden])
            res = tf.matmul(flat_inputs, W)
            if use_bias:
                b = tf.get_variable(
                    "b", [hidden], initializer=tf.constant_initializer(0.))
                res = tf.nn.bias_add(res, b)
            res = tf.reshape(res, out_shape)
            return res
    def dot_attention(self,inputs, memory,scope="dot_attention"):
        with tf.variable_scope(scope):
            inputs_ = tf.nn.relu(self.dense(inputs,self.hidden_size,use_bias=False, scope="inputs"))
            memory_ = tf.nn.relu(self.dense(memory,self.hidden_size,use_bias=False, scope="memory"))
            outputs = tf.matmul(inputs_, tf.transpose(memory_, [0, 2, 1])) / (self.hidden_size** 0.5)
            logits = tf.nn.softmax(outputs,axis=-1)
            outputs = tf.matmul(logits,memory)
            res = tf.concat([inputs,outputs], axis=2)
        with tf.variable_scope("dot_att_gate"):
            dim = res.get_shape().as_list()[-1]
            d_res=tf.nn.dropout(res,keep_prob=self.dropout_keep_prob)
            gate = tf.nn.sigmoid(self.dense(d_res, dim, use_bias=False))
            return res * gate
    def _match(self):
        with tf.variable_scope("att_encoder"):
            qc_att=self.dot_attention(self.sep_p_encodes,self.sep_q_encodes)
            att,_=rnn('bi-gru', inputs=qc_att, length=self.p_length, hidden_size=self.hidden_size,dropout_keep_prob=self.dropout_keep_prob)
        with tf.variable_scope("match"):
            self_att=self.dot_attention(att,att)
            self.match_p_encodes,_=rnn('bi-gru',inputs=self_att,length=self.p_length,hidden_size=self.hidden_size,dropout_keep_prob=self.dropout_keep_prob)
            
        
    
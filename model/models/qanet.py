#coding=utf-8
'''
Created on Jan 9, 2019

@author: lyk
'''
import tensorflow as tf
import math
import time
import numpy as np
from tensorflow.python.ops import nn_ops
from model.basemodel.rc_model import RCModel as basemodel
initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)
class QANET(basemodel):
    def __init__(self,vocab,args):
        super(QANET,self).__init__(vocab,args)
    def get_timing_signal_1d(self,length, channels, min_timescale=1.0, max_timescale=1.0e4):
        """Gets a bunch of sinusoids of different frequencies.
        Each channel of the input Tensor is incremented by a sinusoid of a different
        frequency and phase.
        This allows attention to learn to use absolute and relative positions.
        Timing signals should be added to some precursors of both the query and the
        memory inputs to attention.
        The use of relative position is possible because sin(x+y) and cos(x+y) can be
        experessed in terms of y, sin(x) and cos(x).
        In particular, we use a geometric sequence of timescales starting with
        min_timescale and ending with max_timescale.  The number of different
        timescales is equal to channels / 2. For each timescale, we
        generate the two sinusoidal signals sin(timestep/timescale) and
        cos(timestep/timescale).  All of these sinusoids are concatenated in
        the channels dimension.
        Args:
        length: scalar, length of timing signal sequence.
        channels: scalar, size of timing embeddings to create. The number of
            different timescales is equal to channels / 2.
        min_timescale: a float
        max_timescale: a float
        Returns:
        a Tensor of timing signals [1, length, channels]
        """
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
                (tf.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])
        return signal
    def add_timing_signal_1d(self,x, min_timescale=1.0, max_timescale=1.0e4):
        """Adds a bunch of sinusoids of different frequencies to a Tensor.
        Each channel of the input Tensor is incremented by a sinusoid of a different
        frequency and phase.
        This allows attention to learn to use absolute and relative positions.
        Timing signals should be added to some precursors of both the query and the
        memory inputs to attention.
        The use of relative position is possible because sin(x+y) and cos(x+y) can be
        experessed in terms of y, sin(x) and cos(x).
        In particular, we use a geometric sequence of timescales starting with
        min_timescale and ending with max_timescale.  The number of different
        timescales is equal to channels / 2. For each timescale, we
        generate the two sinusoidal signals sin(timestep/timescale) and
        cos(timestep/timescale).  All of these sinusoids are concatenated in
        the channels dimension.
        Args:
        x: a Tensor with shape [batch, length, channels]
        min_timescale: a float
        max_timescale: a float
        Returns:
        a Tensor the same shape as x.
        """
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        signal = self.get_timing_signal_1d(length, channels, min_timescale, max_timescale)
        return x + signal
    def layer_norm_compute_python(self,x, epsilon, scale, bias):
        """Layer norm raw computation."""
        mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * scale + bias

    def norm_fn(self,x, filters=None, epsilon=1e-6, scope=None, reuse=None):
        """Layer normalize the tensor x, averaging over the last dimension."""
        if filters is None:
            filters = x.get_shape()[-1]
        with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
            scale = tf.get_variable(
                "layer_norm_scale", [filters], regularizer = regularizer, initializer=tf.ones_initializer())
            bias = tf.get_variable(
                "layer_norm_bias", [filters], regularizer = regularizer, initializer=tf.zeros_initializer())
            result =self.layer_norm_compute_python(x, epsilon, scale, bias)
            return result
    def depthwise_separable_convolution(self,inputs, kernel_size, num_filters,
                                    scope = "depthwise_separable_convolution",
                                    bias = True, is_training = True, reuse = None):
        with tf.variable_scope(scope, reuse = reuse):
            shapes = inputs.shape.as_list()
            depthwise_filter = tf.get_variable("depthwise_filter",
                                            (kernel_size[0], kernel_size[1], shapes[-1], 1),
                                            dtype = tf.float32,
                                            regularizer=regularizer,
                                            initializer = initializer_relu())
            pointwise_filter = tf.get_variable("pointwise_filter",
                                            (1,1,shapes[-1],num_filters),
                                            dtype = tf.float32,
                                            regularizer=regularizer,
                                            initializer = initializer_relu())
            outputs = tf.nn.separable_conv2d(inputs,
                                            depthwise_filter,
                                            pointwise_filter,
                                            strides = (1,1,1,1),
                                            padding = "SAME")
            if bias:
                b = tf.get_variable("bias",
                        outputs.shape[-1],
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())
                outputs += b
            outputs = tf.nn.relu(outputs)
            return outputs
    def layer_dropout(self,inputs, residual, dropout):
        pred = tf.random_uniform([]) < dropout
        return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs,dropout) + residual)
    def conv_block(self,inputs, num_conv_layers, kernel_size, num_filters,seq_len = None, scope = "conv_block", is_training = True,
               reuse = None, bias = True, dropout = 0.0, sublayers = (1, 1)):
        with tf.variable_scope(scope, reuse = reuse):
            outputs = tf.expand_dims(inputs,2)
            l, L = sublayers
            for i in range(num_conv_layers):
                residual = outputs
                outputs = self.norm_fn(outputs, scope = "layer_norm_%d"%i, reuse = reuse)
                if (i) % 2 == 0:
                    outputs = tf.nn.dropout(outputs,dropout)
                outputs = self.depthwise_separable_convolution(outputs,kernel_size = (kernel_size, 1), num_filters = num_filters,
                    scope = "depthwise_conv_layers_%d"%i, is_training = is_training, reuse = reuse)
                outputs = self.layer_dropout(outputs, residual, dropout * float(l) / L)
                l += 1
            return tf.squeeze(outputs,2), l
    def split_last_dimension(self,x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = x.get_shape().dims
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
        ret.set_shape(new_shape)
        return tf.transpose(ret,[0,2,1,3])
    def mask_logits(self,inputs, mask, mask_value = -1e30):
        shapes = inputs.shape.as_list()
        mask = tf.cast(mask, tf.float32)
        return inputs + mask_value * (1 - mask)
    def dot_product_attention(self,q,
                          k,
                          v,
                          bias,
                          seq_len = None,
                          mask = None,
                          is_training = True,
                          scope=None,
                          reuse = None,
                          dropout = 0.0):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        with tf.variable_scope(scope, default_name="dot_product_attention", reuse = reuse):
            # [batch, num_heads, query_length, memory_length]
            logits = tf.matmul(q, k, transpose_b=True)
            if bias:
                b = tf.get_variable("bias",
                        logits.shape[-1],
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())
                logits += b
            if mask is not None:
                shapes = [x  if x != None else -1 for x in logits.shape.as_list()]
                mask = tf.reshape(mask, [shapes[0],1,1,shapes[-1]])
                logits = self.mask_logits(logits, mask)
            weights = tf.nn.softmax(logits, name="attention_weights")
            # dropping out the attention links for each of the heads
            weights = tf.nn.dropout(weights,dropout)
            return tf.matmul(weights, v)
    def combine_last_two_dimensions(self,x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        ret.set_shape(new_shape)
        return ret
    def multihead_attention(self,queries, units, num_heads,
                        memory = None,
                        seq_len = None,
                        scope = "Multi_Head_Attention",
                        reuse = None,
                        mask = None,
                        is_training = True,
                        bias = True,
                        dropout = 0.0):
        with tf.variable_scope(scope, reuse = reuse):
            # Self attention
            if memory is None:
                memory = queries
    
            memory = self.conv(memory, 2 * units, name = "memory_projection", reuse = reuse)
            query = self.conv(queries, units, name = "query_projection", reuse = reuse)
            Q = self.split_last_dimension(query, num_heads)
            K, V = [self.split_last_dimension(tensor, num_heads) for tensor in tf.split(memory,2,axis = 2)]
    
            key_depth_per_head = units // num_heads
            Q *= key_depth_per_head**-0.5
            x = self.dot_product_attention(Q,K,V,
                                      bias = bias,
                                      seq_len = seq_len,
                                      mask = mask,
                                      is_training = is_training,
                                      scope = "dot_product_attention",
                                      reuse = reuse, dropout = dropout)
            return self.combine_last_two_dimensions(tf.transpose(x,[0,2,1,3]))
    def self_attention_block(self,inputs, num_filters, seq_len, mask = None, num_heads = 8,
                         scope = "self_attention_ffn", reuse = None, is_training = True,
                         bias = True, dropout = 0.0, sublayers = (1, 1)):
        with tf.variable_scope(scope, reuse = reuse):
            l, L = sublayers
            # Self attention
            outputs = self.norm_fn(inputs, scope = "layer_norm_1", reuse = reuse)
            outputs = tf.nn.dropout(outputs,dropout)
            outputs = self.multihead_attention(outputs, num_filters,num_heads = num_heads, seq_len = seq_len, reuse = reuse,
                mask = mask, is_training = is_training, bias = bias, dropout = dropout)
            residual = self.layer_dropout(outputs, inputs, dropout * float(l) / L)
            l += 1
            # Feed-forward
            outputs = self.norm_fn(residual, scope = "layer_norm_2", reuse = reuse)
            outputs = tf.nn.dropout(outputs,dropout)
            outputs = self.conv(outputs, num_filters, True, tf.nn.relu, name = "FFN_1", reuse = reuse)
            outputs = self.conv(outputs, num_filters, True, None, name = "FFN_2", reuse = reuse)
            outputs = self.layer_dropout(outputs, residual, dropout * float(l) / L)
            l += 1
            return outputs, l
    def residual_block(self,inputs, num_blocks, num_conv_layers, kernel_size, mask = None,num_filters = 128, input_projection = False, num_heads = 8,seq_len = None, scope = "res_block", is_training = True,reuse = None, bias = True, dropout = 0.0):
        with tf.variable_scope(scope, reuse = reuse):
            if input_projection:
                inputs = self.conv(inputs, num_filters, name = "input_projection", reuse = reuse)
            outputs = inputs
            sublayer = 1
            total_sublayers = (num_conv_layers + 2) * num_blocks
            for i in range(num_blocks):
                outputs = self.add_timing_signal_1d(outputs)
                outputs, sublayer = self.conv_block(outputs, num_conv_layers, kernel_size, num_filters,
                    seq_len = seq_len, scope = "encoder_block_%d"%i,reuse = reuse, bias = bias,
                    dropout = dropout, sublayers = (sublayer, total_sublayers))
                outputs, sublayer = self.self_attention_block(outputs, num_filters, seq_len, mask = mask, num_heads = num_heads,
                    scope = "self_attention_layers%d"%i, reuse = reuse, is_training = is_training,
                    bias = bias, dropout = dropout, sublayers = (sublayer, total_sublayers))
            return outputs
    def conv(self,inputs, output_size, bias = None, activation = None, kernel_size = 1, name = "conv", reuse = None):
        with tf.variable_scope(name, reuse = reuse):
            shapes = inputs.shape.as_list()
            if len(shapes) > 4:
                raise NotImplementedError
            elif len(shapes) == 4:
                filter_shape = [1,kernel_size,shapes[-1],output_size]
                bias_shape = [1,1,1,output_size]
                strides = [1,1,1,1]
            else:
                filter_shape = [kernel_size,shapes[-1],output_size]
                bias_shape = [1,1,output_size]
                strides = 1
            conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
            kernel_ = tf.get_variable("kernel_",
                            filter_shape,
                            dtype = tf.float32,
                            regularizer=regularizer,
                            initializer = initializer_relu() if activation is not None else initializer())
            outputs = conv_func(inputs, kernel_, strides, "VALID")
            if bias:
                outputs += tf.get_variable("bias_",
                            bias_shape,
                            regularizer=regularizer,
                            initializer = tf.zeros_initializer())
            if activation is not None:
                return activation(outputs)
            else:
                return outputs

    def highway(self,x, size = None, activation = None,num_layers = 2, scope = "highway", dropout = 0.0, reuse = None):
        with tf.variable_scope(scope, reuse):
            if size is None:
                size = x.shape.as_list()[-1]
            else:
                x = self.conv(x, size, name = "input_projection", reuse = reuse)
            for i in range(num_layers):
                T = self.conv(x, size, bias = True, activation = tf.sigmoid,
                         name = "gate_%d"%i, reuse = reuse)
                H = self.conv(x, size, bias = True, activation = activation,
                         name = "activation_%d"%i, reuse = reuse)
                H = tf.nn.dropout(H,dropout)
                x = H * T + x * (1.0 - T)
            return x
    def ndim(self,x):
        """Copied from keras==2.0.6
        Returns the number of axes in a tensor, as an integer.
    
        # Arguments
            x: Tensor or variable.
    
        # Returns
            Integer (scalar), number of axes.
    
        # Examples
        ```python
            >>> from keras import backend as K
            >>> inputs = K.placeholder(shape=(2, 4, 5))
            >>> val = np.array([[1, 2], [3, 4]])
            >>> kvar = K.variable(value=val)
            >>> K.ndim(inputs)
            3
            >>> K.ndim(kvar)
            2
        ```
        """
        dims = x.get_shape()._dims
        if dims is not None:
            return len(dims)
        return None
    def dot(self,x, y):
        """Modified from keras==2.0.6
        Multiplies 2 tensors (and/or variables) and returns a *tensor*.
    
        When attempting to multiply a nD tensor
        with a nD tensor, it reproduces the Theano behavior.
        (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)
    
        # Arguments
            x: Tensor or variable.
            y: Tensor or variable.
    
        # Returns
            A tensor, dot product of `x` and `y`.
        """
        if self.ndim(x) is not None and (self.ndim(x) > 2 or self.ndim(y) > 2):
            x_shape = []
            for i, s in zip(x.get_shape().as_list(), tf.unstack(tf.shape(x))):
                if i is not None:
                    x_shape.append(i)
                else:
                    x_shape.append(s)
            x_shape = tuple(x_shape)
            y_shape = []
            for i, s in zip(y.get_shape().as_list(), tf.unstack(tf.shape(y))):
                if i is not None:
                    y_shape.append(i)
                else:
                    y_shape.append(s)
            y_shape = tuple(y_shape)
            y_permute_dim = list(range(self.ndim(y)))
            y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
            xt = tf.reshape(x, [-1, x_shape[-1]])
            yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
            return tf.reshape(tf.matmul(xt, yt),
                              x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
        if isinstance(x, tf.SparseTensor):
            out = tf.sparse_tensor_dense_matmul(x, y)
        else:
            out = tf.matmul(x, y)
        return out
    def batch_dot(self,x, y, axes=None):
        """Copy from keras==2.0.6
        Batchwise dot product.
    
        `batch_dot` is used to compute dot product of `x` and `y` when
        `x` and `y` are data in batch, i.e. in a shape of
        `(batch_size, :)`.
        `batch_dot` results in a tensor or variable with less dimensions
        than the input. If the number of dimensions is reduced to 1,
        we use `expand_dims` to make sure that ndim is at least 2.
    
        # Arguments
            x: Keras tensor or variable with `ndim >= 2`.
            y: Keras tensor or variable with `ndim >= 2`.
            axes: list of (or single) int with target dimensions.
                The lengths of `axes[0]` and `axes[1]` should be the same.
    
        # Returns
            A tensor with shape equal to the concatenation of `x`'s shape
            (less the dimension that was summed over) and `y`'s shape
            (less the batch dimension and the dimension that was summed over).
            If the final rank is 1, we reshape it to `(batch_size, 1)`.
        """
        if isinstance(axes, int):
            axes = (axes, axes)
        x_ndim = self.ndim(x)
        y_ndim = self.ndim(y)
        if x_ndim > y_ndim:
            diff = x_ndim - y_ndim
            y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
        elif y_ndim > x_ndim:
            diff = y_ndim - x_ndim
            x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
        else:
            diff = 0
        if self.ndim(x) == 2 and self.ndim(y) == 2:
            if axes[0] == axes[1]:
                out = tf.reduce_sum(tf.multiply(x, y), axes[0])
            else:
                out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
        else:
            if axes is not None:
                adj_x = None if axes[0] == self.ndim(x) - 1 else True
                adj_y = True if axes[1] == self.ndim(y) - 1 else None
            else:
                adj_x = None
                adj_y = None
            out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
        if diff:
            if x_ndim > y_ndim:
                idx = x_ndim + y_ndim - 3
            else:
                idx = x_ndim - 1
            out = tf.squeeze(out, list(range(idx, idx + diff)))
        if self.ndim(out) == 1:
            out = tf.expand_dims(out, 1)
        return out
    def optimized_trilinear_for_attention(self,args, c_maxlen, q_maxlen, input_keep_prob=1.0,
        scope='efficient_trilinear',
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=initializer()):
        assert len(args) == 2, "just use for computing attention with two input"
        arg0_shape = args[0].get_shape().as_list()
        arg1_shape = args[1].get_shape().as_list()
        if len(arg0_shape) != 3 or len(arg1_shape) != 3:
            raise ValueError("`args` must be 3 dims (batch_size, len, dimension)")
        if arg0_shape[2] != arg1_shape[2]:
            raise ValueError("the last dimension of `args` must equal")
        arg_size = arg0_shape[2]
        dtype = args[0].dtype
        droped_args = [tf.nn.dropout(arg, input_keep_prob) for arg in args]
        with tf.variable_scope(scope):
            weights4arg0 = tf.get_variable(
                "linear_kernel4arg0", [arg_size, 1],
                dtype=dtype,
                regularizer=regularizer,
                initializer=kernel_initializer)
            weights4arg1 = tf.get_variable(
                "linear_kernel4arg1", [arg_size, 1],
                dtype=dtype,
                regularizer=regularizer,
                initializer=kernel_initializer)
            weights4mlu = tf.get_variable(
                "linear_kernel4mul", [1, 1, arg_size],
                dtype=dtype,
                regularizer=regularizer,
                initializer=kernel_initializer)
            biases = tf.get_variable(
                "linear_bias", [1],
                dtype=dtype,
                regularizer=regularizer,
                initializer=bias_initializer)
            subres0 = tf.tile(self.dot(droped_args[0], weights4arg0), [1, 1, q_maxlen])
            subres1 = tf.tile(tf.transpose(self.dot(droped_args[1], weights4arg1), perm=(0, 2, 1)), [1, c_maxlen, 1])
            subres2 = self.batch_dot(droped_args[0] * weights4mlu, tf.transpose(droped_args[1], perm=(0, 2, 1)))
            res = subres0 + subres1 + subres2
            nn_ops.bias_add(res, biases)
            return res
    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()#using bi-lstm passage question encoded
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))
    def _encode(self):
        self.c_maxlen=tf.reduce_max(self.p_length)
        self.q_maxlen=tf.reduce_max(self.q_length)
        p_emb=self.highway(self.p_emb, size=self.hidden_size,scope="highway", dropout=self.dropout_keep_prob, reuse=None)
        q_emb=self.highway(self.q_emb, size=self.hidden_size,scope="highway", dropout=self.dropout_keep_prob, reuse=True)
        with tf.variable_scope("Embedding_Encoder_Layer"):
            c = self.residual_block(p_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = None,
                num_filters = self.hidden_size,
                num_heads = 1,
                seq_len = None,
                scope = "Encoder_Residual_Block",
                bias = False,
                dropout = self.dropout_keep_prob)
            q = self.residual_block(q_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = None,
                num_filters = self.hidden_size,
                num_heads = 1,
                seq_len = None,
                scope = "Encoder_Residual_Block",
                reuse = True, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout_keep_prob)
        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            S = self.optimized_trilinear_for_attention([c, q], self.c_maxlen, self.q_maxlen, input_keep_prob = self.dropout_keep_prob)
            S_=tf.nn.softmax(logits=S, axis=-1)#p*q
            S_T=tf.transpose(tf.nn.softmax(logits=S, axis=1),(0,2,1))#q*p
            self.c2q=tf.matmul(S_,q)#p*q * q*d=p*d
            self.q2c=tf.matmul(tf.matmul(S_,S_T),c)#p*q * q*p * p*d=p*d
            attention_outputs = [c,self.c2q,c*self.c2q,c*self.q2c]#[p*d,p*d,p*d,p*d]
        self.enc=list()
        with tf.variable_scope("Model_Encoder_Layer"):
            inputs=tf.concat(attention_outputs,axis=-1)
            self.enc = [self.conv(inputs,self.hidden_size,name="input_projection")]
            for i in range(3):
                if i % 2 ==0:
                    self.enc[i] = tf.nn.dropout(self.enc[i],self.dropout_keep_prob)
                self.enc.append(
                    self.residual_block(self.enc[i],
                        num_blocks = 7,
                        num_conv_layers = 2,
                        kernel_size = 5,
                        mask = None,
                        num_filters = self.hidden_size,
                        num_heads = 1,
                        seq_len = None,
                        scope = "Model_Encoder",
                        bias = False,
                        reuse = True if i > 0 else None,
                        dropout = self.dropout_keep_prob)
                    )
        with tf.variable_scope("Output_Layer"):
            batch_size = tf.shape(self.start_label)[0]
            s_enc=tf.concat([self.enc[1], self.enc[2]],axis = -1)
            s_enc=tf.reshape(s_enc,shape=[batch_size,-1,s_enc.get_shape().as_list()[-1]])
            start_logits=tf.squeeze(self.conv(s_enc,1, bias = False, name = "start_pointer"),-1)
            e_enc=tf.concat([self.enc[1], self.enc[3]],axis = -1)
            e_enc=tf.reshape(e_enc,shape=[batch_size,-1,e_enc.get_shape().as_list()[-1]])
            end_logits=tf.squeeze(self.conv(e_enc,1, bias = False, name = "end_pointer"), -1)
            self.start_probs = tf.nn.softmax(start_logits,axis=-1)
            self.end_probs = tf.nn.softmax(end_logits,axis=-1)
    def _compute_loss(self):
        """
        The loss function
        """

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses

        self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss
            
            
            
            
        
        
    
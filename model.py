import tensorflow as tf
from tensorflow.contrib import rnn

class textrnn:
    def __init__(self, classes, sentence_length, embed_size, target_is_prob, word_size,
                 is_sigmoid_loss=True, learning_rate=1e-3, optimizer=tf.train.AdamOptimizer, word_embeding=None, **kwargs):
        '''
        :param classes: class labels count.
        :param sentence_length
        :param embed_size: word embeding dimension.
        :param target_is_prob: wheather taget ouput is prob or 0/1 labels
        :param word_size
        :param word_embeding: pretrained_word_embeding
        :param is_sigmoid_loss: simoid_loss or softmax_loss
        :param kwargs:
        '''
        self.word_size = word_size
        self.embed_size = embed_size
        self.input = tf.placeholder(tf.int32, shape=(None, sentence_length)) # feed word_index sequence
        self.target = tf.placeholder(tf.float32, shape=(None, sentence_length, classes))
        self.sentence_size = sentence_length
        self.classes = classes
        self.is_sigmoid_loss = is_sigmoid_loss
        self.learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32, name="learning_rate")
        self.optimizer = optimizer(self.learning_rate)
        if word_embeding is None:
            raise Exception("word embeding could not be None!")
        self.load_embeding(word_embeding)
        if "activation" not in kwargs:
            self.activation = "relu"
        else:
            self.activation = kwargs["activation"]
        if "regulazier" not in kwargs:
            self.regulazier = "l2"
        else:
            self.regulazier = kwargs["regulazier"]
        if "rnn_units" not in kwargs:
            self.rnn_units = 128
        else:
            self.rnn_units = int(kwargs["rnn_units"])
        if "fc_units" not in kwargs:
            self.fc_units = 400
        else:
            self.fc_units = int(kwargs["fc_units"])
        if "fc_kinit" not in kwargs:
            self.fc_kinit = tf.truncated_normal_initializer(stddev=0.1)
        else:
            # fc_kinit should be an instance of tf class
            self.fc_kinit = kwargs["fc_kinit"]
        if "l2_lambda" not in kwargs:
            self.l2_lambda = 0.0001
        else:
            self.l2_lambda = kwargs["l2_lambda"]
        self.define_graph()
        self.define_loss()
        self.define_optimizer()

    def init_weight(self):
        with tf.variable_scope("init_weight"):
            pass

    def load_embeding(self, embeding):
        with tf.variable_scope("load_embeding"):
            self.word_embeding = tf.get_variable("word_embeding",shape=(self.word_size, self.embed_size),
                                            dtype=tf.float32, initializer=tf.constant_initializer(embeding))
    def lstm(self):
        with tf.variable_scope("lstm_Cell"):
            lstm_cell = rnn.BasicLSTMCell(self.rnn_units, reuse=True)
        return lstm_cell

    def define_graph(self):
        with tf.variable_scope("embeding_lookup"):
            features = tf.nn.embedding_lookup(self.word_embeding, self.input)
        with tf.variable_scope("bilstm_layer_1"):
            cell_fw = [self.lstm() for i in range(self.sentence_size)]
            cell_bw = [self.lstm() for i in range(self.sentence_size)]
            h_states,_,_ = rnn.stack_bidirectional_rnn(cell_fw, cell_bw, features)
        with tf.variable_scope("bilstm_layer_2"):
            cell_fw = [self.lstm() for i in range(self.sentence_size)]
            cell_bw = [self.lstm() for i in range(self.sentence_size)]
            h_states, _, _ = rnn.stack_bidirectional_rnn(cell_fw, cell_bw, h_states) # (batch_size, sentence_size, num_units*2)
        with tf.variable_scope("pooling_layer"):
            maxpool_layer = tf.layers.max_pooling1d(pool_size=self.sentence_size, strides=1, padding="valid")(h_states)
            avepool_layer = tf.layers.average_pooling1d(pool_size=self.sentence_size, strides=1, padding="valid")(h_states)
            # (batch_size, num_units*2)
        with tf.variable_scope("concat_layer"):
            concat_layer = tf.concat([maxpool_layer, avepool_layer], axis=1)
        with tf.variable_scope("fc_layer"):
            fc_layer = tf.layers.dense(concat_layer, self.fc_units, kernel_initializer=self.fc_kinit,
                                        activation=tf.nn.relu)
            self.logits = tf.layers.dense(fc_layer, self.classes, kernel_initializer=self.fc_kinit,
                                         activation=None)
        with tf.varible_scope("output_layer"):
            self.output = tf.sigmoid(self.logits) #TODO

    def define_loss(self):
        if self.is_sigmoid_loss:
            with tf.variable_scope("compute_loss"):
                self.losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target, logits=self.logits)
                self.loss = tf.reduce_mean(self.losses)  + \
                            tf.add_n([tf.nn.l2_loss(v)  for v in tf.trainable_variables() if "bias" not in v.name]) * self.l2_lambda
        else:
            with tf.variable_scope("compute_loss"):
                self.losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits)
                self.loss = tf.reduce_mean(self.losses) + \
                            tf.add_n([tf.nn.l2_loss(v)  for v in tf.trainable_variables() if "bias" not in v.name]) * self.l2_lambda

    def define_optimizer(self):
        with tf.variable_scope("optimize"):
            train_vars = tf.trainable_variables()
            grads = self.optimizer.compute_gradients(self.loss, var_list=train_vars)
            self.trainop = self.optimizer.apply_gradients(grads)
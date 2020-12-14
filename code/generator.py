import tensorflow as tf


class Generator():
    def __init__(self, n_node, node_emd_init, config):
        self.n_node = n_node
        self.emd_dim = config.n_emb
        self.node_emd_init = node_emd_init

        #with tf.variable_scope('generator'):
        if node_emd_init:
            self.node_embedding_matrix = tf.get_variable(name = 'gen_node_embedding',
                                                       shape = self.node_emd_init.shape,
                                                       initializer = tf.constant_initializer(self.node_emd_init),
                                                       trainable = True)
        else:
            self.node_embedding_matrix = tf.get_variable(name = 'gen_node_embedding',
                                                       shape = [self.n_node, self.emd_dim],
                                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                                       trainable = True)

        self.gen_w_1 = tf.get_variable(name = 'gen_w',
                                       shape = [2, self.emd_dim, self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)
        self.gen_b_1 = tf.get_variable(name = 'gen_b',
                                       shape = [2, self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)
        self.gen_w_2 = tf.get_variable(name = 'gen_w_2',
                                       shape = [2, self.emd_dim, self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)
        self.gen_b_2 = tf.get_variable(name = 'gen_b_2',
                                       shape = [2, self.emd_dim],
                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                       trainable = True)

        self.node_ids = tf.placeholder(tf.int32, shape = [None])

        self.noise_embedding = tf.placeholder(tf.float32, shape = [2, None, self.emd_dim])

        self.dis_node_embedding = tf.placeholder(tf.float32, shape = [2, None, self.emd_dim])
        
        self.node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.node_ids)

        _noise_embedding= []
        for i in range(2):
            _noise_embedding.append(tf.reshape(tf.nn.embedding_lookup(self.noise_embedding, tf.constant([i])), 
                                               [-1, self.emd_dim]))
        _dis_node_embedding = []
        for i in range(2):
            _dis_node_embedding.append(tf.reshape(tf.nn.embedding_lookup(self.dis_node_embedding, tf.constant([i])), 
                                                  [-1, self.emd_dim]))

        _neg_loss = [0.0, 0.0]
        _fake_node_embedding_list = []
        _score = [0, 0]
        
        
        for i in range(2):
            _fake_node_embedding = self.generate_node(self.node_embedding, _noise_embedding[i], i)
            _fake_node_embedding_list.append(_fake_node_embedding)

            _score[i] = tf.reduce_sum(tf.multiply(_dis_node_embedding[i], _fake_node_embedding), axis=1)

            _neg_loss[i] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                            labels=tf.ones_like(_score[i]) * (1.0 - config.label_smooth), logits=_score[i])) \
                            + config.lambda_gen * (tf.nn.l2_loss(self.node_embedding) + tf.nn.l2_loss(self.gen_w_1[i]))

        self.neg_loss = _neg_loss
        self.fake_node_embedding = _fake_node_embedding_list
        self.loss = self.neg_loss[0] + self.neg_loss[1]

        optimizer = tf.train.AdamOptimizer(config.lr_gen)
        #optimizer = tf.train.RMSPropOptimizer(config.lr_gen)
        self.g_updates = optimizer.minimize(self.loss)

    def generate_node(self, node_embedding, noise_embedding, direction):
        #node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, node_id)
        #relation_embedding = tf.nn.embedding_lookup(self.relation_embedding_matrix, relation_id)

        input = tf.reshape(node_embedding, [-1, self.emd_dim])
        #input = tf.concat([input, noise_embedding], axis = 1)
        input = input + noise_embedding

        output = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1[direction]) + self.gen_b_1[direction])
        #input = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_1) + self.gen_b_1)# +  relation_embedding
        #output = tf.nn.leaky_relu(tf.matmul(input, self.gen_w_2) + self.gen_b_2)
        #output = node_embedding + relation_embedding + noise_embedding

        return output

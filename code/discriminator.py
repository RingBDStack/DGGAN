import tensorflow as tf


class Discriminator():
    def __init__(self, n_node, node_emd_init, config):
        self.n_node = n_node
        self.emd_dim = config.n_emb
        self.node_emd_init = node_emd_init

        #with tf.variable_scope('disciminator'):
        if node_emd_init:
            self.node_embedding_matrix = tf.get_variable(name = 'dis_node_embedding',
                                                         shape = self.node_emd_init.shape,
                                                         initializer = tf.constant_initializer(self.node_emd_init),
                                                         trainable = True)
        else:
            self.node_embedding_matrix = tf.get_variable(name = 'dis_node_embedding',
                                                         shape = [2, self.n_node, self.emd_dim],
                                                         initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                                         trainable = True)

        self.pos_node_ids = tf.placeholder(tf.int32, shape = [None])

        self.pos_node_neighbor_ids = tf.placeholder(tf.int32, shape = [None])

        self.fake_node_embedding = tf.placeholder(tf.float32, shape = [2, 2, None, self.emd_dim])

        _node_embedding_matrix = []
        for i in range(2):
            _node_embedding_matrix.append(tf.reshape(tf.nn.embedding_lookup(self.node_embedding_matrix, tf.constant([i])), [-1, self.emd_dim]))

        self.pos_node_embedding = tf.nn.embedding_lookup(_node_embedding_matrix[0], self.pos_node_ids)
        self.pos_node_neighbor_embedding = tf.nn.embedding_lookup(_node_embedding_matrix[1], self.pos_node_neighbor_ids)
        pos_score = tf.matmul(self.pos_node_embedding, self.pos_node_neighbor_embedding, transpose_b=True)
        self.pos_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_score), logits=pos_score))
        
        _neg_loss = [0, 0, 0, 0]
        node_id = [self.pos_node_ids, self.pos_node_neighbor_ids]
        for i in range(2):
            for j in range(2):
                node_embedding = tf.nn.embedding_lookup(_node_embedding_matrix[j], node_id[i])
                _fake_node_embedding = tf.reshape(tf.nn.embedding_lookup(self.fake_node_embedding, tf.constant([i])), [2, -1, self.emd_dim])
                _fake_node_embedding = tf.reshape(tf.nn.embedding_lookup(_fake_node_embedding, tf.constant([j])), [-1, self.emd_dim])
                neg_score = tf.matmul(node_embedding, _fake_node_embedding, transpose_b=True)
                _neg_loss[i * 2 + j] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_score), logits=neg_score))

        self.neg_loss = _neg_loss
        self.loss = self.pos_loss + self.neg_loss[0] * config.neg_weight[0] + self.neg_loss[1] * config.neg_weight[1] + \
                self.neg_loss[2] * config.neg_weight[2] + self.neg_loss[3] * config.neg_weight[3]

        optimizer = tf.train.AdamOptimizer(config.lr_dis)
        #optimizer = tf.train.GradientDescentOptimizer(config.lr_dis)
        #optimizer = tf.train.RMSPropOptimizer(config.lr_dis)
        self.d_updates = optimizer.minimize(self.loss)
        #self.reward = tf.log(1 + tf.exp(tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)))

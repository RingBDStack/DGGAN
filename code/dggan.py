import os
import tensorflow as tf
import time
import numpy as np
import random
import math
import utils
import config
import evaluation
from generator import Generator
from discriminator import Discriminator
import warnings
warnings.filterwarnings('ignore')


class Model():
    def __init__(self):
        t = time.time()
        print('reading graph...')
        self.graph, self.n_node, self.node_list, self.node_list_s, self.egs = utils.read_graph(config.train_file)
        self.node_emd_shape = [2, self.n_node, config.n_emb]
        print('[%.2f] reading graph finished. #node = %d' % (time.time() - t, self.n_node))

        self.dis_node_embed_init = None
        self.gen_node_embed_init = None
        if config.pretrain_dis_node_emb:
            t = time.time()
            print('reading initial embeddings...')
            dis_node_embed_init = np.array([utils.read_embeddings(filename=x, n_node=self.n_node, n_embed=config.n_emb) \
                                            for x in [config.pretrain_dis_node_emb]])
            gen_node_embed_init = np.array([utils.read_embeddings(filename=x, n_node=self.n_node, n_embed=config.n_emb) \
                                            for x in [config.pretrain_gen_node_emb]])
            print('[%.2f] read initial embeddings finished.' % (time.time() - t))

        print('building DGGAN model...')
        self.discriminator = None
        self.generator = None
        self.build_generator()
        self.build_discriminator()
        if config.experiment == 'link_prediction':
            self.link_prediction = evaluation.LinkPrediction(config)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = self.config)
        self.saver = tf.train.Saver(max_to_keep=0)
        if config.pretrain_ckpt:
            print('restore...')
            pretrain_ckpt = tf.train.latest_checkpoint(config.pretrain_ckpt)
            self.saver.restore(self.sess, pretrain_ckpt)
        else:
            print('initial...')
            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.sess.run(self.init_op)

    def build_discriminator(self):
        #with tf.variable_scope("discriminator"):
        self.discriminator = Discriminator(n_node = self.n_node,
                                           node_emd_init = self.dis_node_embed_init,
                                           config = config)
    def build_generator(self):
        #with tf.variable_scope("generator"):
        self.generator = Generator(n_node = self.n_node,
                                   node_emd_init = self.gen_node_embed_init,
                                   config = config)

    def train_dis(self, dis_loss, pos_loss, neg_loss, dis_cnt):
        np.random.shuffle(self.egs)
        
        info = ''
        for index in range(math.floor(len(self.egs) / config.d_batch_size)):
            pos_node_ids, pos_node_neighbor_ids, fake_node_embedding = self.prepare_data_for_d(index, self.egs)
            _, _loss, _pos_loss, _neg_loss = self.sess.run([self.discriminator.d_updates, self.discriminator.loss, \
                                            self.discriminator.pos_loss, self.discriminator.neg_loss],
                                            feed_dict = {self.discriminator.pos_node_ids : np.array(pos_node_ids),
                                                         self.discriminator.pos_node_neighbor_ids : np.array(pos_node_neighbor_ids),
                                                         self.discriminator.fake_node_embedding : np.array(fake_node_embedding)})

            dis_loss += _loss
            pos_loss += _pos_loss
            for i in range(4):
                neg_loss[i] += _neg_loss[i]
            dis_cnt += 1
            info = 'dis_loss=%.4f pos_loss=%.4f neg_loss_0=%.4f neg_loss_1=%.4f neg_loss_2=%.4f neg_loss_3=%.4f' % \
                (dis_loss / dis_cnt, pos_loss / dis_cnt, neg_loss[0] / dis_cnt, neg_loss[1] / dis_cnt, \
                 neg_loss[2] / dis_cnt, neg_loss[3] / dis_cnt)
            self.my_print(info, True, 1)
        return (dis_loss, pos_loss, neg_loss, dis_cnt)

    def train_gen(self, gen_loss, neg_loss, gen_cnt):
        np.random.shuffle(self.node_list)
        
        info = ''
        for index in range(math.floor(len(self.node_list) / config.g_batch_size)):
            node_ids, noise_embedding, dis_node_embedding = self.prepare_data_for_g(index, self.node_list)
            _, _loss, _neg_loss = self.sess.run([self.generator.g_updates, self.generator.loss, self.generator.neg_loss],
                                                 feed_dict = {self.generator.node_ids : np.array(node_ids),
                                                              self.generator.noise_embedding : np.array(noise_embedding),
                                                              self.generator.dis_node_embedding : np.array(dis_node_embedding)})

            gen_loss += _loss
            for i in range(2):
                neg_loss[i] += _neg_loss[i]
            gen_cnt += 1
            info = 'gen_loss=%.4f neg_loss_0=%.4f neg_loss_1=%.4f' % (gen_loss / gen_cnt, neg_loss[0] / gen_cnt, neg_loss[1] / gen_cnt)
            self.my_print(info, True, 1)
        return (gen_loss, neg_loss, gen_cnt)

    def train(self):
        best_auc = [[0]*3, [0]*3, [0]*3]
        best_epoch = [-1, -1, -1]

        print('start traning...')
        for epoch in range(config.n_epoch):
            info = 'epoch %d' % epoch
            self.my_print(info, False, 1)

            dis_loss = 0.0
            dis_pos_loss = 0.0
            dis_neg_loss = [0.0, 0.0, 0.0, 0.0]
            dis_cnt = 0

            gen_loss = 0.0
            gen_neg_loss = [0.0, 0.0]
            gen_cnt = 0

            #D-step
            for d_epoch in range(config.d_epoch):
                dis_loss, dis_pos_loss, dis_neg_loss, dis_cnt = self.train_dis(dis_loss, dis_pos_loss, dis_neg_loss, dis_cnt)
                self.my_print('', False, 1)
                auc = self.evaluate()

            #G-step
            for g_epoch in range(config.g_epoch):
                gen_loss, gen_neg_loss, gen_cnt = self.train_gen(gen_loss, gen_neg_loss, gen_cnt)
                self.my_print('', False, 1)

            if config.save:
                self.write_embeddings_to_file(epoch)
        print('training finished.')

    def prepare_data_for_d(self, index, egs):
        pos_node_ids = []
        pos_node_neighbor_ids = []

        for eg in egs[index * config.d_batch_size : (index + 1) * config.d_batch_size]:
            node_id, node_neighbor_id = eg

            pos_node_ids.append(node_id)
            pos_node_neighbor_ids.append(node_neighbor_id)

        # generate fake node
        fake_node_embedding = []

        noise_embedding = np.random.normal(0.0, config.sig, (2, len(pos_node_ids), config.n_emb))
        fake_node_embedding.append(self.sess.run(self.generator.fake_node_embedding,
                                            feed_dict = {self.generator.node_ids : np.array(pos_node_ids),
                                                         self.generator.noise_embedding : np.array(noise_embedding)}))

        noise_embedding = np.random.normal(0.0, config.sig, (2, len(pos_node_ids), config.n_emb))
        fake_node_embedding.append(self.sess.run(self.generator.fake_node_embedding,
                                            feed_dict = {self.generator.node_ids : np.array(pos_node_neighbor_ids),
                                                         self.generator.noise_embedding : np.array(noise_embedding)}))

        return pos_node_ids, pos_node_neighbor_ids, fake_node_embedding

    def prepare_data_for_g(self, index, node_list):
        node_ids = []

        for node_id in node_list[index * config.g_batch_size : (index + 1) * config.g_batch_size]:
            node_ids.append(node_id)

        noise_embedding = np.random.normal(0.0, config.sig, (2, len(node_ids), config.n_emb))

        dis_node_embedding = []
        dis_node_embedding1 = self.sess.run([self.discriminator.pos_node_embedding],
                                             feed_dict = {self.discriminator.pos_node_ids : np.array(node_ids)})
        dis_node_embedding2 = self.sess.run([self.discriminator.pos_node_neighbor_embedding],
                                             feed_dict = {self.discriminator.pos_node_neighbor_ids : np.array(node_ids)})
        dis_node_embedding = np.vstack([dis_node_embedding1, dis_node_embedding2])
        return node_ids, noise_embedding, dis_node_embedding
    
    def evaluate(self):
        if config.experiment == 'link_prediction':
            return self.evaluate_link_prediction()
    
    def evaluate_link_prediction(self):
        embedding_matrix = self.sess.run(self.discriminator.node_embedding_matrix)
        auc = self.link_prediction.evaluate(embedding_matrix)
        info = 'auc_0=%.4f auc_50=%.4f auc_100=%.4f' % (auc[0], auc[1], auc[2])
        self.my_print(info, False, 1)
        return auc

    def write_embeddings_to_file(self, epoch):
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        models = [self.generator, self.discriminator]
        emb_filenames = ['gen.emb', 'dis_s.emb', 'dis_t.emb']
        embedding_matrix = [self.sess.run(self.generator.node_embedding_matrix)]
        embedding_matrix.extend([self.sess.run(self.discriminator.node_embedding_matrix)[0],
                                 self.sess.run(self.discriminator.node_embedding_matrix)[1]])
        for i in range(3):
            index = np.array(range(self.n_node)).reshape(-1, 1)
            t = np.hstack([index, embedding_matrix[i]])
            embedding_list = t.tolist()
            embedding_str = [str(int(emb[0])) + ' ' + ' '.join([str(x) for x in emb[1:]]) + '\n' for emb in embedding_list]

            file_path = '%s%d-%s' % (config.save_path, epoch, emb_filenames[i])
            with open(file_path, 'w') as f:
                lines = [str(self.n_node) + ' ' + str(config.n_emb) + '\n'] + embedding_str
                f.writelines(lines)
        self.saver.save(self.sess, config.save_path + 'model.ckpt', global_step=epoch)
    
    def my_print(self, info, r_flag, verbose):
        if verbose == 1 and config.verbose == 0:
            return
        if r_flag:
            print('\r%s' % info, end='')
        else:
            print('%s' % info)


if __name__ == '__main__':
    model = Model()
    model.train()

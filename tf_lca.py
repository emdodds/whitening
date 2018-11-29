# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
Created on Wed Jan 25 13:53:03 2017

@author: Eric
"""

import numpy as np
import tensorflow as tf
import tf_sparsenet
# workaround for cluster issue
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('Failed to import matplotlib, plotting unavailable.')


class LCALearner(tf_sparsenet.Sparsenet):

    def __init__(self,
                 data,
                 datatype="image",
                 pca=None,
                 nunits=200,
                 batch_size=100,
                 paramfile='dummy',
                 moving_avg_rate=0.01,
                 stimshape=None,
                 lam=0.15,
                 niter=200,
                 infrate=0.1,
                 learnrate=50.0,
                 snr_goal=None,
                 seek_snr_rate=0.1,
                 threshfunc='hard',
                 store_every=1):
        """
        Sparse dictionary learner using the
        L1 or L0 locally competitive algorithm
        from Rozell et al 2008 for inference.
        Parameters
        data        : [nsamples, ndim] numpy array of training data
        datatype    : (str) "image" or "spectro"
        pca         : PCA object with inverse_transform(), or None
        nunits      : (int) number of units in sparsenet model
        batch_size  : (int) number of samples in each batch for learning
        paramfile   : (str) filename for pickle file storing parameters
        moving_avg_rate: (float) rate for updating average statistics
        stimshape   : (array-like) original shape of each training datum
        lam         : (float) sparsity parameter; higher means more sparse
        niter       : (int) number of time steps in inference (not adjustable)
        infrate     : (float) rate for inference
        learnrate   : (float) gradient descent rate for learning
                      loss gets divided by numinput, so make this larger
        snr_goal    : (float) snr in dB, lam adjusted dynamically to match
        seek_snr_rate : (float) rate parameter for adjusting lam as above
        threshfunc  : (str) specifies which thresholding function to use
        """
        # save input parameters
        self.nunits = nunits
        self.batch_size = batch_size
        self.paramfile = paramfile
        self.moving_avg_rate = moving_avg_rate
        self.stimshape = stimshape or ((16, 16) if datatype == 'image'
                                       else (25, 256))
        self.infrate = infrate
        self.niter = niter
        self.learnrate = learnrate
        self.threshfunc = threshfunc
        self.lam = lam
        self.snr_goal = snr_goal
        self.seek_snr_rate = seek_snr_rate
        self.store_every = store_every

        # initialize model
        self._load_stims(data, datatype, self.stimshape, pca)
        self.Q = tf.random_normal([self.nunits, self.stims.datasize])
        self.graph = self.build_graph()
        self.initialize_stats()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.config = tf.ConfigProto(gpu_options=gpu_options)
        self.config.gpu_options.allow_growth = True
        with tf.Session(graph=self.graph, config=self.config) as sess:
            sess.run(self._init_op)
            self.Q = sess.run(self.renorm_phi)

    def acts(self, uu, ll):
        """Computes the activation function given the internal varaiable uu
        and the current threshold parameter ll."""
        if self.threshfunc.startswith('hard'):
            thresholded = tf.identity
        else:
            def thresholded(prethresh):
                return tf.add(prethresh, -tf.sign(prethresh)*ll)

        if self.threshfunc.endswith('pos') or self.threshfunc.endswith('rec'):
            rect = tf.identity
        else:
            rect = tf.abs

        return tf.where(tf.greater(rect(uu), ll),
                        thresholded(uu), tf.multiply(0.0, uu),
                        name='activity')

    def build_graph(self):
        graph = tf.get_default_graph()

        self._infrate = tf.Variable(self.infrate, trainable=False)
        self._learnrate = tf.Variable(self.learnrate, trainable=False)
        self.thresh = tf.Variable(self.lam, trainable=False)

        self.phi = tf.Variable(self.Q)

        self.x = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.stims.datasize])

        self.final_acts = tf.Variable(tf.zeros([self.nunits, self.batch_size]))

        # LCA inference
        self.lca_drive = tf.matmul(self.phi, tf.transpose(self.x))
        self.lca_gram = (tf.matmul(self.phi, tf.transpose(self.phi)) -
                         tf.constant(np.identity(int(self.nunits)),
                         dtype=np.float32))

        def next_u(old_u_l, ii):
            old_u = old_u_l[0]
            ll = old_u_l[1]
            lca_compet = tf.matmul(self.lca_gram, self.acts(old_u, ll))
            du = self.lca_drive - lca_compet - old_u
            new_l = tf.constant(0.98)*ll  # 0.98 lifted from Bruno's code
            new_l = tf.where(tf.greater(new_l, self.thresh),
                             new_l,
                             self.thresh*np.ones(self.batch_size))
            return (old_u + self.infrate*du, new_l)

        self._itercount = tf.constant(np.arange(self.niter))
        init_u_l = (tf.zeros([self.nunits, self.batch_size]),
                    0.5*tf.reduce_max(tf.abs(self.lca_drive), axis=0))
        self._inftraj = tf.scan(next_u, self._itercount, initializer=init_u_l)
        self._infu = self._inftraj[0]
        self._infl = self._inftraj[1]
        self.u = self._infu[-1]

        # for testing inference
        self._infacts = self.acts(self._infu, tf.expand_dims(self._infl, 1))

        def mul_fn(someacts):
            return tf.matmul(tf.transpose(someacts), self.phi)
        self._infxhat = tf.map_fn(mul_fn, self._infacts)
        self._infresid = self.x - self._infxhat
        self._infmse = tf.reduce_sum(tf.square(self._infresid), axis=[1, 2])
        self._infmse = self._infmse/self.batch_size/self.stims.datasize

        self.full_inference = self.final_acts.assign(self.acts(self.u,
                                                               self.thresh))
        self.xhat = tf.matmul(tf.transpose(self.final_acts), self.phi)
        self.resid = self.x - self.xhat
        self.mse = tf.reduce_sum(tf.square(self.resid))
        self.mse = self.mse/self.batch_size/self.stims.datasize
        self.meanL1 = tf.reduce_sum(tf.abs(self.final_acts))/self.batch_size
        self.loss = 0.5*self.mse

        learner = tf.train.GradientDescentOptimizer(self.learnrate)
        self.learn_op = learner.minimize(self.loss,
                                         var_list=[self.phi])

        self.renorm_phi = self.phi.assign(tf.nn.l2_normalize(self.phi, axis=1))

        self.snr = tf.reduce_mean(tf.square(self.x))/self.mse
        if self.snr_goal is not None:
            convert = tf.constant(np.log(10.0)/10.0, dtype=tf.float32)
            snr_ratio = self.snr/tf.exp(convert*tf.constant(self.snr_goal,
                                                            dtype=tf.float32))
            self.seek_snr = self.thresh.assign(self.thresh *
                                               tf.pow(snr_ratio,
                                                      self.seek_snr_rate))
        self.snr_db = 10.0*tf.log(self.snr)/np.log(10.0)

        self._init_op = tf.global_variables_initializer()

        return graph

    def train_step(self, sess):
        feed_dict = {self.x: self.get_batch()}

        # run inference and get activities
        acts = sess.run(self.full_inference, feed_dict=feed_dict)

        # get losses and update weights
        # update lambda to seek reconstruction snr if specified
        op_list = [self.mse, self.meanL1, self.learn_op]
        if self.snr_goal is not None:
            op_list += [self.seek_snr]
            mse_value, meanL1_value, _, _ = sess.run(op_list,
                                                     feed_dict=feed_dict)
        else:
            mse_value, meanL1_value, _ = sess.run(op_list,
                                                  feed_dict=feed_dict)

        # normalize the weights
        sess.run(self.renorm_phi)

        return acts, 0.5*mse_value, mse_value, meanL1_value

    def infer(self, x):
        """Return activities for given data.
        x.shape[0] should be a multiple of batch_size."""
        with tf.Session(graph=self.graph, config=self.config) as sess:
            self.initialize_vars(sess)
            nexamples = x.shape[0]
            nbatches = nexamples/self.batch_size
            if int(nbatches) != nbatches:
                raise ValueError("Use a multiple of the batch size.")
            nbatches = int(nbatches)
            acts = np.zeros((self.nunits, 0))
            for ii in range(nbatches):
                newx = x[ii*self.batch_size:(ii+1)*self.batch_size]
                feed_dict = {self.x: newx}
                newacts = sess.run([self.full_inference], feed_dict=feed_dict)
                acts = np.concatenate([acts]+newacts, axis=1)
                if nbatches > 1 and ii % 50 == 0:
                    print("Completed {} batches of {}".format(ii+1, nbatches))
        return acts

    def initialize_vars(self, sess):
        sess.run(self._init_op)
        sess.run([self.phi.assign(self.Q),
                  self.thresh.assign(self.lam),
                  self._infrate.assign(self.infrate),
                  self._learnrate.assign(self.learnrate),
                  self.thresh.assign(self.lam)])

    def retrieve_vars(self, sess):
        """Retrieve values from tf graph."""
        stuff = sess.run([self.phi,
                          self._infrate,
                          self._learnrate,
                          self.thresh])
        self.Q, self.infrate, self.learnrate, self.lam = stuff

    def test_inference(self, x=None):
        if x is None:
            feed_dict = {self.x: self.get_batch()}
        else:
            feed_dict = {self.x: x}
        with tf.Session(graph=self.graph, config=self.config) as sess:
            self.initialize_vars(sess)
            acts, costs = sess.run([self.full_inference,
                                    self._infmse],
                                   feed_dict=feed_dict)
            snr = sess.run(self.snr_db, feed_dict=feed_dict)
        plt.plot(costs, 'b')
        print("Final SNR: " + str(snr))
        return acts, costs

    def get_param_list(self):
        return {'nunits': self.nunits,
                'batch_size': self.batch_size,
                'paramfile': self.paramfile,
                'lam': self.lam,
                'niter': self.niter,
                'infrate': self.learnrate,
                'learnrate': self.infrate}


class LCALearnerTI(LCALearner):
    """LCALearner with backprop through inference.
    Currently only supports soft threshold."""
    def build_graph(self):
        graph = tf.get_default_graph()

        self._infrate = tf.Variable(self.infrate, trainable=False)
        self._learnrate = tf.Variable(self.learnrate, trainable=False)
        self.thresh = tf.Variable(self.lam, trainable=False)

        self.phi = tf.Variable(self.Q)

        self.x = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.stims.datasize])

        # LCA inference
        self.lca_drive = tf.matmul(self.phi, tf.transpose(self.x))
        self.lca_gram = (tf.matmul(self.phi, tf.transpose(self.phi)) -
                         tf.constant(np.identity(int(self.nunits)),
                         dtype=np.float32))

        def next_u(old_u_l, ii):
            old_u = old_u_l[0]
            ll = old_u_l[1]
            lca_compet = tf.matmul(self.lca_gram, self.acts(old_u, ll))
            du = self.lca_drive - lca_compet - old_u
            new_l = tf.constant(0.98)*ll  # 0.98 lifted from Bruno's code
            new_l = tf.where(tf.greater(new_l, self.thresh),
                             new_l,
                             self.thresh*np.ones(self.batch_size))
            return (old_u + self.infrate*du, new_l)

        self._itercount = tf.constant(np.arange(self.niter))
        init_u_l = (tf.zeros([self.nunits, self.batch_size]),
                    0.5*tf.reduce_max(tf.abs(self.lca_drive), axis=0))
        self._inftraj = tf.scan(next_u, self._itercount, initializer=init_u_l)
        self._infu = self._inftraj[0]
        self._infl = self._inftraj[1]
        self.u = self._infu[-1]

        # for testing inference
        self._infacts = self.acts(self._infu, tf.expand_dims(self._infl, 1))

        def mul_fn(someacts):
            return tf.matmul(tf.transpose(someacts), self.phi)
        self._infxhat = tf.map_fn(mul_fn, self._infacts)
        self._infresid = self.x - self._infxhat
        self._infmse = tf.reduce_sum(tf.square(self._infresid), axis=[1, 2])
        self._infmse = self._infmse/self.batch_size/self.stims.datasize

        self.final_acts = self.acts(self.u, self.thresh)
        self.xhat = tf.matmul(tf.transpose(self.final_acts), self.phi)
        self.resid = self.x - self.xhat
        self.mse = tf.reduce_sum(tf.square(self.resid))
        self.mse = self.mse/self.batch_size/self.stims.datasize
        self.meanL1 = tf.reduce_sum(tf.abs(self.final_acts))/self.batch_size
        # with gradients evaluated through a/u, we need to keep the sparsity
        self.loss = 0.5*self.mse + self.lam*self.meanL1/self.stims.datasize

        learner = tf.train.GradientDescentOptimizer(self.learnrate)
        self.learn_op = learner.minimize(self.loss,
                                         var_list=[self.phi])

        self.renorm_phi = self.phi.assign(tf.nn.l2_normalize(self.phi, dim=1))

        self.snr = tf.reduce_mean(tf.square(self.x))/self.mse
        if self.snr_goal is not None:
            convert = tf.constant(np.log(10.0)/10.0, dtype=tf.float32)
            snr_ratio = self.snr/tf.exp(convert*tf.constant(self.snr_goal,
                                                            dtype=tf.float32))
            self.seek_snr = self.thresh.assign(self.thresh *
                                               tf.pow(snr_ratio,
                                                      self.seek_snr_rate))
        self.snr_db = 10.0*tf.log(self.snr)/np.log(10.0)

        return graph

    def test_inference(self, x=None):
        if x is None:
            feed_dict = {self.x: self.get_batch()}
        else:
            feed_dict = {self.x: x}
        with tf.Session(graph=self.graph, config=self.config) as sess:
            self.initialize_vars(sess)
            acts, costs = sess.run([self.final_acts,
                                    self._infmse],
                                   feed_dict=feed_dict)
            snr = sess.run(self.snr_db, feed_dict=feed_dict)
        plt.plot(costs, 'b')
        print("Final SNR: " + str(snr))
        return acts, costs

    def train_step(self, sess):
        feed_dict = {self.x: self.get_batch()}

        op_list = [self.final_acts, self.mse, self.meanL1, self.learn_op,
                   self.seek_snr]
        acts, mse_value, meanL1_value, _, _ = sess.run(op_list,
                                                       feed_dict=feed_dict)

        # normalize the weights
        sess.run(self.renorm_phi)

        return acts, 0.5*mse_value, mse_value, meanL1_value

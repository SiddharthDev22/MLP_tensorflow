import tensorflow as tf
from ops import *


class NeuralNet:
    # Class properties
    __network = None         # Graph for Network
    __train_op = None        # Operation used to optimize loss function
    __loss = None            # Loss function to be optimized, which is based on predictions
    __accuracy = None        # Classification accuracy
    __probs = None           # Prediction probability matrix of shape [batch_size, numClasses]

    def __init__(self, numClass, inputSize):
        self.inputSize = inputSize
        self.numClass = numClass
        self.h1 = 70		        # Number of neurons in the first fully-connected layer
        self.h2 = 15		        # Number of neurons in the second fully-connected layer
        self.init_lr = 0.00005	    # Initial learning rate

        self.x, self.y, self.keep_prob = self.create_placeholders()

    def create_placeholders(self):
        with tf.name_scope('Input'):
            x = tf.placeholder(tf.float32,
                               shape=(None, self.inputSize),
                               name='x-input')
            y = tf.placeholder(tf.float32,
                               shape=(None, self.numClass),
                               name='y-input')
            keep_prob = tf.placeholder(tf.float32)
        return x, y, keep_prob

    def inference(self):
        if self.__network:
            return self
        # Building network...
        with tf.variable_scope('NeuralNet'):
            net = fc_layer(self.x, self.h1, 'FC1',
                           add_reg=False,
                           nonlinearity='relu',
                           batch_normalize=True)
            net = dropout(net, self.keep_prob)
            net = fc_layer(net, self.h2, 'FC2',
                           add_reg=False,
                           nonlinearity='relu',
                           batch_normalize=True)
            net = dropout(net, self.keep_prob)
            net = fc_layer(net, self.numClass, 'FC3', add_reg=False)
            self.__network = net
        return self

    def pred_func(self):
        if self.__probs:
            return self
        self.__probs = tf.nn.softmax(self.__network)
        return self

    def accuracy_func(self):
        if self.__accuracy:
            return self
        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.__network, 1), tf.argmax(self.y, 1))
            self.__accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.__accuracy)
        return self

    def loss_func(self):
        if self.__loss:
            return self
        with tf.name_scope('Loss'):
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.__network)
            self.__loss = tf.reduce_mean(diff)
            tf.summary.scalar('cross_entropy', self.__loss)
        return self

    def train_func(self):
        if self.__train_op:
            return self
        with tf.name_scope('Train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.init_lr)
            self.__train_op = optimizer.minimize(self.__loss)
        return self

    @property
    def network(self):
        return self.__network

    @property
    def probs(self):
        return self.__probs

    @property
    def train_op(self):
        return self.__train_op

    @property
    def loss(self):
        return self.__loss

    @property
    def accuracy(self):
        return self.__accuracy
import tensorflow as tf


def weight_variable(name, shape):
    """Create a weight variable with appropriate initialization."""
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name, dtype=tf.float32,
                           shape=shape, initializer=initer)


def bias_variable(name, shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name, dtype=tf.float32,
                           initializer=initial)


def fc_layer(bottom, out_dim, name, add_reg=False, use_relu=True):
    """Create a fully connected layer"""
    in_dim = bottom.get_shape()[1]
    with tf.variable_scope(name):
        weights = weight_variable(name, shape=[in_dim, out_dim])
        tf.summary.histogram('histogram', weights)
        biases = bias_variable(name, [out_dim])
        layer = tf.matmul(bottom, weights)
        layer += biases
        if use_relu:
            layer = tf.nn.relu(layer)
        if add_reg:
            tf.add_to_collection('weights', weights)
    return layer


def flatten_layer(layer):
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat

def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)
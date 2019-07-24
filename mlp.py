"""Multi-Layer Perceptron (MLP) implementation in Python.

One step of back propagation is manually implemented from scratch using Numpy.
Only supports 2 layers (linear, relu) and 1 loss (mse)."""

import numpy as np
import tensorflow as tf

def linear(x, w, b):
    return np.dot(x, w) + b

def linear_grad(do, x, w, b):
    dx = np.dot(do, w.T)
    dw = np.expand_dims(do, axis=1) * np.expand_dims(x, axis=-1)
    dw = np.sum(dw, axis=0)
    db = np.sum(do, axis=0)
    return dx, dw, db 

def relu(x):
    return np.clip(x, 0, x)

def relu_grad(do, x):
    return do * (x > 0).astype(np.float32)

def mse(yhat, y):
    return np.mean((np.squeeze(yhat) - y)**2)

def mse_grad(yhat, y):
    return 2 *  np.expand_dims(np.squeeze(yhat) - y, axis=-1) / yhat.shape[0]


if __name__ == '__main__':
    # Features and labels.
    x = np.random.uniform(size=(5, 2))  # 5 examples, 100 features.
    y = np.random.uniform(size=(5,))  # 5 (random) labels.

    # Layers.
    w1 = np.random.uniform(size=(2, 2))
    b1 = np.random.uniform(size=(2,))
    w2 = np.random.uniform(size=(2, 2))
    b2 = np.random.uniform(size=(2,))
    w3 = np.random.uniform(size=(2, 1))
    b3 = np.random.uniform(size=(1,))

    # Forward.
    x2 = linear(x, w1, b1)
    r2 = relu(x2)
    x3 = linear(r2, w2, b2)
    r3 = relu(x3)
    yhat = linear(r3, w3, b3)
    loss = mse(yhat, y)

    # Backward.
    dyhat = mse_grad(yhat, y)
    dx3, dw3, db3 = linear_grad(dyhat, x3, w3, b3)
    dr3 = relu_grad(dx3, x3)
    dx2, dw2, db2 = linear_grad(dr3, x2, w2, b2)
    dr2 = relu_grad(dx2, x2)
    dx, dw1, db1 = linear_grad(dx2, x, w1, b1)

    # Get gradients in TF and check that they are close.
    tf_x = tf.constant(x)
    tf_y = tf.constant(y)

    tf_w1 = tf.get_variable('w1', initializer=w1)
    tf_b1 = tf.get_variable('b1', initializer=b1)
    tf_w2 = tf.get_variable('w2', initializer=w2)
    tf_b2 = tf.get_variable('b2', initializer=b2)
    tf_w3 = tf.get_variable('w3', initializer=w3)
    tf_b3 = tf.get_variable('b3', initializer=b3)

    tf_x2 = tf.nn.relu(tf.matmul(tf_x, tf_w1) + tf_b1)
    tf_x3 = tf.nn.relu(tf.matmul(tf_x2, tf_w2) + tf_b2)
    tf_yhat = tf.nn.relu(tf.matmul(tf_x3, tf_w3) + tf_b3)
    tf_loss = tf.reduce_mean(tf.square(tf.squeeze(tf_yhat) - tf_y))
    gradients = tf.gradients(
            tf_loss, [tf_yhat, tf_b3, tf_w3, tf_b2, tf_w2, tf_b1, tf_w1]) 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Sanity check that the weights and biases are equal at init.
        eps = 1e-12
        assert np.all(np.abs(x - sess.run(tf_x)) < eps)
        assert np.all(np.abs(y - sess.run(tf_y)) < eps)
        assert np.all(np.abs(w1 - sess.run(tf_w1)) < eps)
        assert np.all(np.abs(b1 - sess.run(tf_b1)) < eps)
        assert np.all(np.abs(w2 - sess.run(tf_w2)) < eps)
        assert np.all(np.abs(b2 - sess.run(tf_b2)) < eps)
        assert np.all(np.abs(w3 - sess.run(tf_w3)) < eps)
        assert np.all(np.abs(b3 - sess.run(tf_b3)) < eps)

        # Sanity check that the loss is equal.
        assert np.abs(loss - sess.run(tf_loss)) < eps

        # Check that the gradients are equal.
        tf_dyhat, tf_db3, tf_dw3, tf_db2, tf_dw2, tf_db1, tf_dw1 = sess.run(gradients)
        assert np.all(np.abs(dyhat - tf_dyhat) < eps)
        assert np.all(np.abs(db3 - tf_db3) < eps)
        assert np.all(np.abs(dw3 - tf_dw3) < eps)
        assert np.all(np.abs(db2 - tf_db2) < eps)
        assert np.all(np.abs(dw2 - tf_dw2) < eps)
        assert np.all(np.abs(db1 - tf_db1) < eps)
        assert np.all(np.abs(dw1 - tf_dw1) < eps)




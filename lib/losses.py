"""
Losses personalizadas
"""
import tensorflow as tf
import numpy as np
def EMD_loss(y_true, y_pred):
    """
    Earth mover's distance, para usar al compilar un modelo
    """
    n = np.prod(y_true.shape)
    p = tf.math.subtract(y_true, y_pred)
    p = tf.math.square(p)
    p = tf.math.reduce_sum(p)
    return tf.math.sqrt(tf.math.divide(p,n))
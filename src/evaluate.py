import tensorflow as tf
import numpy as np

def rmspe_tf(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    percentage_error = (y_true - y_pred) / (y_true + 1e-7)
    return tf.sqrt(tf.reduce_mean(tf.square(percentage_error)))

def rmspe_np(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + 1e-7))))

if __name__ == "__main__":
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([90.0, 210.0, 310.0])
    print("rmspe_np test:", rmspe_np(y_true, y_pred))

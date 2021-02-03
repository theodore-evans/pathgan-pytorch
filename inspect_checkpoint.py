#%%
import os
from pprint import pprint
import tensorflow as tf

tf_path = os.path.abspath('./checkpoint/PathologyGAN.ckt')  # Path to our TensorFlow checkpoint
tf_vars = tf.train.list_variables(tf_path)
print(tf_vars)
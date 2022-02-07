import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


## Defining functions for network layers 
def conv2d( inputs , filters , stride_size, bias ):
    out = tf.nn.conv2d( inputs , filters , strides=[ 1 , stride_size , stride_size , 1 ] , padding="SAME" )
    out = tf.nn.bias_add(out, bias) 
    return out

def maxpool( inputs , pool_size , stride_size ):
    return tf.nn.max_pool2d( inputs , ksize=[ 1 , pool_size , pool_size , 1 ] , padding='SAME' , strides=[ 1 , stride_size , stride_size , 1 ] )

def dense( inputs , weights, bias ):
    out = tf.matmul( inputs , weights )
    out = tf.nn.bias_add(out, bias)
    return out

def loss1( true , pred ):
    return tf.losses.categorical_crossentropy( true , pred )

def get_weight( shape , name ):
    return tf.Variable( initializer( shape ) , name=name , trainable=True , dtype=tf.float32 )

def get_bias (shape, name):
    return tf.Variable(initializer2 (shape), name = name, trainable = True, dtype = tf.float32)

##Initializers for weights and biases
initializer = tf.initializers.glorot_uniform(seed = 42)
initializer2 = tf.initializers.Constant(0.0)

shapes = [
    [3, 3, 3, 64],
    [3, 3, 64, 128],
    [3, 3, 128, 256],
    [4096, 128],
    [128, 32],
    [32, 10],
    [64],
    [128],
    [256],
    [128],
    [32],
    [10],
]

weights = []
for i in range( 6 ):
    weights.append( get_weight( shapes[ i ] , 'weight{}'.format( i ) ) )
    weights.append(get_bias(shapes[i + 6], "bias{}".format (i+6)))

def model( x, Training ) :
    x = tf.cast( x , dtype=tf.float32 )
    c1 = conv2d( x , weights[ 0 ] , stride_size=1, bias = weights[ 1 ] )
    c1 = tf.nn.relu(c1)    
    if Training:
      c1 = tf.nn.dropout(c1, rate = 0.1)
    p1 = maxpool( c1 , pool_size=2 , stride_size=2 ) 
    c2 = conv2d( p1 , weights[ 2 ] , stride_size=1, bias = weights[ 3 ])
    c2 = tf.nn.relu(c2)
    if Training:  
      c2 = tf.nn.dropout(c2, rate = 0.2)
    p2 = maxpool( c2 , pool_size=2 , stride_size=2 )
    c3 = conv2d( p2 , weights[ 4 ] , stride_size=1,  bias = weights[ 5 ] )
    c3 = tf.nn.relu(c3)
    if Training:
      c3 = tf.nn.dropout(c3, rate = 0.2)
    p3 = maxpool( c3 , pool_size=2 , stride_size=2 ) 
    flatten = tf.reshape( p3 , shape=( tf.shape( p3 )[0] , -1 ))
    d1 = dense( flatten , weights[ 6 ], bias = weights[ 7 ] )
    d1 = tf.nn.relu(d1)
    if Training:
      d1 = tf.nn.dropout(d1, rate = 0.4)
    d2 = dense(d1, weights[ 8 ], bias = weights[ 9 ])
    d2 = tf.nn.relu(d2)
    if Training:
      d2 = tf.nn.dropout(d2, rate = 0.5)
    d3 = dense(d2, weights[10], bias = weights [11 ])
    out = tf.nn.softmax(d3)
    return out


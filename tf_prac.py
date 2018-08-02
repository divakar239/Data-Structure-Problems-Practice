# scratch work

import numpy as np
import tensorflow as tf

def main() -> object:

    #weight and bias matrices
    W = tf.Variable([.1], tf.float32)
    b = tf.Variable([-.1], tf.float32)

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = W*x+b
    a = tf.nn.softmax(z)
    a1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=x)



    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print(sess.run([a, a1], {x:[1,2,3,4], y:[0.2,0.3,0.4,0.6]}))

if __name__ == '__main__':
    main()
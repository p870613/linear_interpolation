from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def layer(input, w_shape, b_shape):
     #random_normal_initializer:generate tensor with normal distribution
     weight_stddev = (2.0/w_shape[0]) **0.5
     w_init = tf.random_normal_initializer(stddev=weight_stddev)
     bias_init = tf.constant_initializer(value=0)

     w = tf.get_variable("w", w_shape, initializer = w_init)
     b = tf.get_variable("b", b_shape, initializer = bias_init)
     
     return tf.nn.relu(tf.matmul(input, w) + b)

def inference(x):
     #2 hidden layer 
     with tf.variable_scope("hidden_1"):
          hidden_1 = layer(x, [784, 256], [256])

     with tf.variable_scope("hidden_2"):
          hidden_2 = layer(hidden_1, [256, 256], [256])

     with tf.variable_scope("output"):
          output = layer(hidden_2, [256, 10], [10])
     return output

def loss(output, y):
     xentropy = xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)  
     loss = tf.reduce_mean(xentropy)
     return loss


if __name__ == '__main__':

     x = tf.placeholder("float", [None, 784])
     y = tf.placeholder("float", [None, 10])
     var_list_opt = [None, None, None, None, None, None]
               
     # load
     

     with tf.Session() as sess:
     #saver=tf.train.import_meta_graph('logistic_logs/model-checkpoint-5500.meta')
     #saver.restore(sess, 'logistic_logs/model-checkpoint-5500')
          
          saver = tf.train.import_meta_graph('logistic_logs/model-checkpoint-5500.meta')
          saver.restore(sess, 'logistic_logs/model-checkpoint-5500')

          var_list_opt = [None, None, None, None, None, None]
          name_2_index = {
             "hidden_1/w:0" : 0,
             "hidden_1/b:0" : 1,
             "hidden_2/w:0" : 2,
             "hidden_2/b:0" : 3,
             "output/w:0" : 4,
             "output/b:0" : 5
          }
          

          for item in tf.trainable_variables():
             if item.name in name_2_index:
               index = name_2_index[item.name]
               var_list_opt[index] = item  

          with tf.variable_scope("mlp_init") as scope:
               output_rand = inference(x)
               cost_rand = loss(output_rand, y)

               scope.reuse_variables()

               var_list_rand = ["hidden_1/w", "hidden_1/b", "hidden_2/w", "hidden_2/b", "output/w", "output/b"]
               var_list_rand = [tf.get_variable(v) for v in var_list_rand]
               
               init_op = tf.variables_initializer(var_list_rand)

               sess.run(init_op)


          with tf.variable_scope("mip_inter") as scope:

               alpha = tf.placeholder("float", [1, 1])
               h1_W_inter = var_list_opt[0] * (1 - alpha) + var_list_rand[0] * (alpha)
               h1_b_inter = var_list_opt[1] * (1 - alpha) + var_list_rand[1] * (alpha)
               h2_W_inter = var_list_opt[2] * (1 - alpha) + var_list_rand[2] * (alpha)
               h2_b_inter = var_list_opt[3] * (1 - alpha) + var_list_rand[3] * (alpha)
               o_W_inter = var_list_opt[4] * (1 - alpha) + var_list_rand[4] * (alpha)
               o_b_inter = var_list_opt[5] * (1 - alpha) + var_list_rand[5] * (alpha)

               h1_inter = tf.nn.relu(tf.matmul(x, h1_W_inter) + h1_b_inter)
               h2_inter = tf.nn.relu(tf.matmul(h1_inter, h2_W_inter) + h2_b_inter)
               o_inter = tf.nn.relu(tf.matmul(h2_inter, o_W_inter) + o_b_inter)

               cost_inter = loss(o_inter, y)
               tf.summary.scalar("interpolated_cost", cost_inter)

          result = []
          for a in np.arange(-2, 2, 0.01):
               feed_dict = {
                         x:mnist.test.images, 
                         y:mnist.test.labels, 
                         alpha : [[a]]
               }

               cost = sess.run([cost_inter], feed_dict = feed_dict)
               result.append(cost)

          
          plt.plot(np.arange(-2, 2, 0.01), result, 'ro', linewidth = 0.5)
          plt.ylabel('Incurred Error')
          plt.xlabel('Alpha')
          plt.savefig('Alpha.png')
          
               

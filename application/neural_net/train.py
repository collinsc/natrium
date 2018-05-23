#!/usr/bin/env python3
"""Basic backpropagation network"""

import argparse
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

class MSD():
    """Absraction layer for the million song data"""
    
    GENRES = {
        "Blues"      : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Country"    : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Electronic" : [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Folk"       : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Jazz"       : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Latin"      : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Metal"      : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "New Age"    : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "Pop"        : [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "Punk"       : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "Rap"        : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "Reggae"     : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "RnB"        : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "Rock"       : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "World"      : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }

    def __init__(self, train, train_label, test, test_label):
        """Takes in the training data and testing data, and processes it to tensorflow friendly data"""
        self.train = np.expand_dims(train["duration"].values, axis=1)
        self.train_label = np.stack(train_label["genre"].apply(self.__translate_label).values)
        self.test = np.expand_dims(test["duration"].values, axis=1)
        self.test_label = np.stack(test_label["genre"].apply(self.__translate_label).values)

        self.__idx = 0;

    def __len__(self):
        """Size of the dataset"""
        return len(self.data)

    def next_batch(self, size):
        """Get's a batch from the training set"""
        set_size = len(self.data)
        if self.__idx > set_size:
            return None
        if self.__idx + size > set_size:
            return self.data[self.__idx:], self.labels[self.__idx:]
        output_data = self.data[self.__idx:(self.__idx+size)]
        output_labels = self.labels[self.__idx:(self.__idx+size)]

#        print(output_data, output_labels)
#        print(np.shape(output_data),np.shape(output_labels))

        
        self.__idx += size
        return output_data, output_labels

    def __translate_label(self, label):
        """Translate a genre label to a hotmax output"""
        return np.array(self.GENRES[label])

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
                  
def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    From tensorflow tutorials: 
    https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard"""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, act=tf.nn.relu):
      """Reusable code for making a simple neural net layer.

      It does a matrix multiply, bias add, and then uses relu to nonlinearize.
      It also sets up name scoping so that the resultant graph is easy to read,
      and adds a number of summary ops.
      """
      # Adding a name scope ensures logical grouping of the layers in the graph.
      # This Variable will hold the state of the weights for the layer
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
      biases = bias_variable([output_dim])
      variable_summaries(biases)
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

def main():
    """Train the network.
    Credit to this tutorial: http://adventuresinmachinelearning.com/python-tensorflow-tutorial/"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_file",
        type=argparse.FileType("r"),
        help="The path of the training data file to read in.")
    parser.add_argument(
        "train_label_file",
        type=argparse.FileType("r"),
        help="The path of the training label file to read in.")
    parser.add_argument(
        "test_file",
        type=argparse.FileType("r"),
        help="The path of the testing data file to read in.")
    parser.add_argument(
        "test_label_file",
        type=argparse.FileType("r"),
        help="The path of the testing label file to read in.")


    args = parser.parse_args()
    print("Commencing...")
    # initialise the variables
    learning_rate = 0.5 
    epochs = 10 
    batch_size = 100 

    #Network inputs and outputs
    x_0 = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])

    #Input layer to hidden layer weights
    x_1 = nn_layer(x_0, 784, 300)
    x_2 = nn_layer(x_1, 300, 10)

    #optmise the network
    x_2_clipped = tf.clip_by_value(x_2, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(label * tf.log(x_2_clipped) + (1 - label) * tf.log(1 - x_2_clipped), axis=1))
    optimiser = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)

    #collect statistics
    correct_prediction = tf.equal(tf.argmax(x_2, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    
    print("Opening data")
    train = pd.read_pickle(args.train_file.name)
    train_label = pd.read_pickle(args.train_label_file.name)
    test = pd.read_pickle(args.test_file.name)
    test_label = pd.read_pickle(args.test_label_file.name)

    print("Processing data")
    training_set = MSD(train, train_label, test, test_label)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter( 'logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test')
        sess.run(tf.global_variables_initializer())

        total_batch = int(len(mnist.train.labels)/batch_size)

        print("Training:")
        print("    batch_size:", batch_size)
        print("    learning_rate:", learning_rate)
        print("    epochs:", epochs)
        for epoch in range(epochs):
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                train_summary, train_entropy = sess.run([merged, cross_entropy], feed_dict={x_0: batch_x, label:  batch_y})
                train_writer.add_summary(train_summary, i)
                test_summary, test_acc = sess.run([merged, accuracy], feed_dict={x_0: mnist.test.images, label: mnist.test.labels})
                print("Entropy at step", i, ":", train_entropy)
                test_writer.add_summary(test_summary, i)
            print("Epoch:", (epoch + 1), "cost:", avg_cost)
        print("Training Finished")

if __name__ == "__main__":
    main()

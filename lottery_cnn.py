#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from PIL import Image,ImageOps
from mnist import extract_images

from scipy import ndimage

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features, [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def ndarraytoimage(_ndarray):
    ndarray_reshape = _ndarray.reshape(28,28)
    ndarray_reshape *= 255.0/ndarray_reshape.max()
    image = Image.fromarray(ndarray_reshape)
    return image
    
def main(unused_argv):
  # Load training and eval data
  mnist = learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  
  """
    test show image
  """
  # counter = 0
  # for data in train_data:
  #     data_reshaped = data
  #     data_reshaped = data_reshaped.reshape(28,28)
  #     data_reshaped *= 255.0/data_reshaped.max()
  #     
  #     np.savetxt('test.out', data_reshaped, delimiter=',')
  #   #   data_reshaped = np.resize(data_reshaped,(280*2,280*2))
  #     
  #     if counter > 10:
  #         image = Image.fromarray(data_reshaped)
  #         image.show()
  #       #   print (train_labels[counter])
  #         break;
  #     counter+=1
  # return
  
  """
    evaluate from james dataset
  """
  window_size = 28
  eval_data_array = []
  eval_labels_array = []
  for batch in range(1,3):
      for index in range(0,10):
          im = Image.open("dataset/{0}_{1}.png".format(batch,index)).convert('L')
          im = im.resize((28,28))
          im = ImageOps.invert(im)
          im = ImageOps.expand(im,border=10,fill='black')
          im = im.resize((28,28))
          
          im = ndimage.binary_dilation(im)
          # convet to np array
          im_narray = np.asarray(im, dtype=np.float32)
          
          # normalize from [0,255] to [0,1]
          im_narray /= 255.0
          
          # resize to 28*28
          im_narray = np.resize(im_narray,(28*28))
        
          # append eval data
          eval_data_array.append(im_narray)
          eval_labels_array.append(index)
      
  # add to eval data
  _eval_data = np.asarray(eval_data_array,dtype=np.float32)
  _eval_labels = np.asarray(eval_labels_array, dtype=np.int32)
  
  # preprocessing image
  _eval_data = np.where(_eval_data < 0.3 ,0 , _eval_data)
  _eval_data = np.where(_eval_data > 0.8 ,1 , _eval_data)
  
  # TODO shink and enlarge image for robust eval
  
  # _eval_data = np.append(_eval_data, eval_data[0:10], axis=0)
  # _eval_labels = np.append(_eval_labels, eval_labels[0:10], axis=0)
  
  eval_labels = _eval_labels
  eval_data = _eval_data
  
  for index in range(len(eval_data)):
      image = ndarraytoimage(eval_data[index])
      image.show(title=str(eval_labels[index]))
      print(eval_labels[index])
      
  # Create the Estimator
  mnist_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  """
    training phase
  """
  # Train the model
  # mnist_classifier.fit(
  #     x=train_data,
  #     y=train_labels,
  #     batch_size=100,
  #     steps=20000,
  #     monitors=[logging_hook])

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }

  print (eval_labels)
  print (repr(eval_labels))
  # Evaluate the model and print results
  eval_results = mnist_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
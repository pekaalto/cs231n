

# # Fast layers
# Making convolution and pooling layers fast can be challenging.
# To spare you the pain, we've provided fast implementations of the forward
# and backward passes for convolution and pooling layers in the file
# `cs231n/fast_layers.py`.
# # coding: utf-8

# # Convolutional Networks
# So far we have worked with deep fully-connected networks, using them to explore different optimization strategies and network architectures. Fully-connected networks are a good testbed for experimentation because they are very computationally efficient, but in practice all state-of-the-art results use convolutional networks instead.
#
# First you will implement several layer types that are used in convolutional networks. You will then use these layers to train a convolutional network on the CIFAR-10 dataset.

# In[ ]:

# As usual, a bit of setup

import os
import sys

_wd = "/Users/paalto/study/cs231n/assignment2"
os.chdir(_wd)
sys.path.append(_wd)

import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# In[ ]:

# Load the (preprocessed) CIFAR10 data.

data = get_CIFAR10_data()
for k, v in data.iteritems():
    print '%s: ' % k, v.shape

# # Convolution: Naive forward pass
# The core of a convolutional network is the convolution
# operation. In the file `cs231n/layers.py`,
# implement the forward pass for the convolution layer
# in the function `conv_forward_naive`.
#
# You don't have to worry too much about efficiency
# at this point; just write the code in whatever way you
# find most clear.
#
# You can test your implementation by running the following:

# In[ ]:

x_shape = (2, 3, 4, 4)
w_shape = (3, 3, 4, 4)
x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
b = np.linspace(-0.1, 0.2, num=3)

conv_param = {'stride': 2, 'pad': 1}
out, _ = conv_forward_naive(x, w, b, conv_param)
correct_out = np.array([[[[[-0.08759809, -0.10987781],
    [-0.18387192, -0.2109216]],
    [[0.21027089, 0.21661097],
        [0.22847626, 0.23004637]],
    [[0.50813986, 0.54309974],
        [0.64082444, 0.67101435]]],
    [[[-0.98053589, -1.03143541],
        [-1.19128892, -1.24695841]],
        [[0.69108355, 0.66880383],
            [0.59480972, 0.56776003]],
        [[2.36270298, 2.36904306],
            [2.38090835, 2.38247847]]]]])

# Compare your output to ours; difference should be around 1e-8
print 'Testing conv_forward_naive'
print 'difference: ', rel_error(out, correct_out)

# # Aside: Image processing via convolutions
#
# As fun way to both check your implementation
# and gain a better understanding of the type of
# operation that convolutional layers can perform,
# we will set up an input containing two images
# and manually set up filters that perform common
# image processing operations (grayscale conversion and edge detection).
# The convolution forward pass will apply these operations
# to each of the input images.
# We can then visualize the results as a sanity check.

# In[ ]:

from scipy.misc import imread, imresize

kitten, puppy = imread('kitten.jpg'), imread('puppy.jpg')
# kitten is wide, and puppy is already square
d = kitten.shape[1] - kitten.shape[0]
kitten_cropped = kitten[:, d / 2:-d / 2, :]

img_size = 200  # Make this smaller if it runs too slow
x = np.zeros((2, 3, img_size, img_size))
x[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1))
x[1, :, :, :] = imresize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))

# Set up a convolutional weights holding 2 filters, each 3x3
w = np.zeros((2, 3, 3, 3))

# The first filter converts the image to grayscale.
# Set up the red, green, and blue channels of the filter.
w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]

# Second filter detects horizontal edges in the blue channel.
w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

# Vector of biases. We don't need any bias for the grayscale
# filter, but for the edge detection filter we want to add 128
# to each output so that nothing is negative.
b = np.array([0, 128])

# Compute the result of convolving each input in x with each filter in w,
# offsetting by b, and storing the results in out.
out, _ = conv_forward_naive(x, w, b, {'stride': 1, 'pad': 1})


def imshow_noax(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')


# Show the original images and the results of the conv operation
plt.subplot(2, 3, 1)
imshow_noax(puppy, normalize=False)
plt.title('Original image')
plt.subplot(2, 3, 2)
imshow_noax(out[0, 0])
plt.title('Grayscale')
plt.subplot(2, 3, 3)
imshow_noax(out[0, 1])
plt.title('Edges')
plt.subplot(2, 3, 4)
imshow_noax(kitten_cropped, normalize=False)
plt.subplot(2, 3, 5)
imshow_noax(out[1, 0])
plt.subplot(2, 3, 6)
imshow_noax(out[1, 1])
plt.show()

# # Convolution: Naive backward pass
# Implement the backward pass for the convolution operation
# in the function `conv_backward_naive` in the file `cs231n/layers.py`.
# Again, you don't need to worry too much about computational efficiency.
#
# When you are done, run the following to check your backward pass
# with a numeric gradient check.

# In[ ]:

filters = 5
x = np.random.randn(4, 3, 5, 5)
w = np.random.randn(filters, 3, 3, 3)
b = np.random.randn(filters, )
dout = np.random.randn(4, filters, 5, 5)
#dout = np.ones((4, filters, 5, 5))
conv_param = {'stride': 1, 'pad': 1}

dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x,
    dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w,
    dout)
db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b,
    dout)

out, cache = conv_forward_naive(x, w, b, conv_param)
dx, dw, db = conv_backward_naive(dout, cache)

# Your errors should be around 1e-9'
print 'Testing conv_backward_naive function'
print 'dx error: ', rel_error(dx, dx_num)
print 'dw error: ', rel_error(dw, dw_num)
print 'db error: ', rel_error(db, db_num)

# # Max pooling: Naive forward
# Implement the forward pass for the max-pooling operation in the function
# `max_pool_forward_naive` in the file `cs231n/layers.py`.
#  Again, don't worry too much about computational efficiency.
#
# Check your implementation by running the following:

# In[ ]:

x_shape = (2, 3, 4, 4)
x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

out, _ = max_pool_forward_naive(x, pool_param)

correct_out = np.array([[[[-0.26315789, -0.24842105],
    [-0.20421053, -0.18947368]],
    [[-0.14526316, -0.13052632],
        [-0.08631579, -0.07157895]],
    [[-0.02736842, -0.01263158],
        [0.03157895, 0.04631579]]],
    [[[0.09052632, 0.10526316],
        [0.14947368, 0.16421053]],
        [[0.20842105, 0.22315789],
            [0.26736842, 0.28210526]],
        [[0.32631579, 0.34105263],
            [0.38526316, 0.4]]]])

# Compare your output with ours. Difference should be around 1e-8.
print 'Testing max_pool_forward_naive function:'
print 'difference: ', rel_error(out, correct_out)

# # Max pooling: Naive backward
# Implement the backward pass for the max-pooling operation in the function
# `max_pool_backward_naive` in the file `cs231n/layers.py`.
# You don't need to worry about computational efficiency.
#
# Check your implementation with numeric gradient checking by running the following:

# In[ ]:

x = np.random.randn(3, 2, 8, 8)
dout = np.random.randn(3, 2, 4, 4)

pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

dx_num = eval_numerical_gradient_array(
    lambda x: max_pool_forward_naive(x, pool_param)[0], x, dout)

out, cache = max_pool_forward_naive(x, pool_param)
dx = max_pool_backward_naive(dout, cache)

# Your error should be around 1e-12
print 'Testing max_pool_backward_naive function:'
print 'dx error: ', rel_error(dx, dx_num)
# The fast convolution implementation depends on a
# Cython extension; to compile it you need to run the following from
# the `cs231n` directory:
# 
# ```bash
# python setup.py build_ext --inplace
# ```
# 
# The API for the fast versions of the convolution
# and pooling layers is exactly the same as the naive versions
# that you implemented above: the forward pass receives data,
# weights, and parameters and produces outputs and a cache object;
# the backward pass recieves upstream derivatives and the cache object
# and produces gradients with respect to the data and weights.
# 
# **NOTE:** The fast implementation for pooling will only perform optimally
# if the pooling regions are non-overlapping and tile the input.
# If these conditions are not met then the fast pooling implementation will
# not be much faster than the naive implementation.
# 
# You can compare the performance of the naive and fast versions of
# these layers by running the following:

# In[ ]:

from cs231n.fast_layers import conv_forward_fast, conv_backward_fast
from time import time

x = np.random.randn(100, 3, 31, 31)
w = np.random.randn(25, 3, 3, 3)
b = np.random.randn(25, )
dout = np.random.randn(100, 25, 16, 16)
conv_param = {'stride': 2, 'pad': 1}

t0 = time()
out_naive, cache_naive = conv_forward_naive(x, w, b, conv_param)
t1 = time()
out_fast, cache_fast = conv_forward_fast(x, w, b, conv_param)
t2 = time()

print 'Testing conv_forward_fast:'
print 'Naive: %fs' % (t1 - t0)
print 'Fast: %fs' % (t2 - t1)
print 'Speedup: %fx' % ((t1 - t0) / (t2 - t1))
print 'Difference: ', rel_error(out_naive, out_fast)

t0 = time()
dx_naive, dw_naive, db_naive = conv_backward_naive(dout, cache_naive)
t1 = time()
dx_fast, dw_fast, db_fast = conv_backward_fast(dout, cache_fast)
t2 = time()

print '\nTesting conv_backward_fast:'
print 'Naive: %fs' % (t1 - t0)
print 'Fast: %fs' % (t2 - t1)
print 'Speedup: %fx' % ((t1 - t0) / (t2 - t1))
print 'dx difference: ', rel_error(dx_naive, dx_fast)
print 'dw difference: ', rel_error(dw_naive, dw_fast)
print 'db difference: ', rel_error(db_naive, db_fast)

# In[ ]:

from cs231n.fast_layers import max_pool_forward_fast, max_pool_backward_fast

x = np.random.randn(100, 3, 32, 32)
dout = np.random.randn(100, 3, 16, 16)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

t0 = time()
out_naive, cache_naive = max_pool_forward_naive(x, pool_param)
t1 = time()
out_fast, cache_fast = max_pool_forward_fast(x, pool_param)
t2 = time()

print 'Testing pool_forward_fast:'
print 'Naive: %fs' % (t1 - t0)
print 'fast: %fs' % (t2 - t1)
print 'speedup: %fx' % ((t1 - t0) / (t2 - t1))
print 'difference: ', rel_error(out_naive, out_fast)

t0 = time()
dx_naive = max_pool_backward_naive(dout, cache_naive)
t1 = time()
dx_fast = max_pool_backward_fast(dout, cache_fast)
t2 = time()

print '\nTesting pool_backward_fast:'
print 'Naive: %fs' % (t1 - t0)
print 'speedup: %fx' % ((t1 - t0) / (t2 - t1))
print 'dx difference: ', rel_error(dx_naive, dx_fast)

# # Convolutional "sandwich" layers
# Previously we introduced the concept of "sandwich" layers that combine multiple
# operations into commonly used patterns. In the file `cs231n/layer_utils.py`
# you will find sandwich layers that implement a few commonly
# used patterns for convolutional networks.

# In[ ]:

from cs231n.layer_utils import conv_relu_pool_forward, conv_relu_pool_backward

x = np.random.randn(2, 3, 16, 16)
w = np.random.randn(3, 3, 3, 3)
b = np.random.randn(3, )
dout = np.random.randn(2, 3, 8, 8)
conv_param = {'stride': 1, 'pad': 1}
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

out, cache = conv_relu_pool_forward(x, w, b, conv_param, pool_param)
dx, dw, db = conv_relu_pool_backward(dout, cache)

dx_num = eval_numerical_gradient_array(
    lambda x: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(
    lambda w: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], w, dout)
db_num = eval_numerical_gradient_array(
    lambda b: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], b, dout)

print 'Testing conv_relu_pool'
print 'dx error: ', rel_error(dx_num, dx)
print 'dw error: ', rel_error(dw_num, dw)
print 'db error: ', rel_error(db_num, db)

# In[ ]:

from cs231n.layer_utils import conv_relu_forward, conv_relu_backward

x = np.random.randn(2, 3, 8, 8)
w = np.random.randn(3, 3, 3, 3)
b = np.random.randn(3, )
dout = np.random.randn(2, 3, 8, 8)
conv_param = {'stride': 1, 'pad': 1}

out, cache = conv_relu_forward(x, w, b, conv_param)
dx, dw, db = conv_relu_backward(dout, cache)

dx_num = eval_numerical_gradient_array(lambda x: conv_relu_forward(x, w, b, conv_param)[0], x, dout)
dw_num = eval_numerical_gradient_array(lambda w: conv_relu_forward(x, w, b, conv_param)[0], w, dout)
db_num = eval_numerical_gradient_array(lambda b: conv_relu_forward(x, w, b, conv_param)[0], b, dout)

print 'Testing conv_relu:'
print 'dx error: ', rel_error(dx_num, dx)
print 'dw error: ', rel_error(dw_num, dw)
print 'db error: ', rel_error(db_num, db)

# # Three-layer ConvNet
# Now that you have implemented all the necessary layers,
# we can put them together into a simple convolutional network.
# 
# Open the file `cs231n/cnn.py` and complete the
# implementation of the `ThreeLayerConvNet` class.
# Run the following cells to help you debug:

# ## Sanity check loss
# After you build a new network, one of the first things you
# should do is sanity check the loss.
# When we use the softmax loss, we expect the loss for random weights
# (and no regularization) to be about
# `log(C)` for `C` classes. When we add regularization this should go up.

# In[ ]:

model = ThreeLayerConvNet()

N = 50
X = np.random.randn(N, 3, 32, 32)
y = np.random.randint(10, size=N)

loss, grads = model.loss(X, y)
print 'Initial loss (no regularization): ', loss

model.reg = 0.5
loss, grads = model.loss(X, y)
print 'Initial loss (with regularization): ', loss

# ## Gradient check
# After the loss looks reasonable, use numeric gradient checking
# to make sure that your backward pass is correct.
# When you use numeric gradient checking you should use a small
# amount of artifical data and a small number of neurons at each layer.

# In[ ]:

num_inputs = 2
input_dim = (3, 16, 16)
reg = 0.0
num_classes = 10
X = np.random.randn(num_inputs, *input_dim)
y = np.random.randint(num_classes, size=num_inputs)

model = ThreeLayerConvNet(num_filters=3, filter_size=3,
    input_dim=input_dim, hidden_dim=7,
    dtype=np.float64)
loss, grads = model.loss(X, y)
for param_name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
    e = rel_error(param_grad_num, grads[param_name])
    print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))


# ## Overfit small data
# A nice trick is to train your model with just
# a few training samples.
# You should be able to overfit small datasets,
# which will result in very high training accuracy
# and comparatively low validation accuracy.

# In[ ]:

num_train = 100
small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'],
    'y_val': data['y_val'],
}

model = ThreeLayerConvNet(weight_scale=1e-2)

solver = Solver(model, small_data,
    num_epochs=20, batch_size=50,
    update_rule='adam',
    optim_config={
        'learning_rate': 1e-4,
    },
    verbose=True, print_every=5000)
solver.train()

# Plotting the loss, training accuracy, and validation accuracy should show clear overfitting:

# In[ ]:

plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# ## Train the net
# By training the three-layer convolutional network for one epoch,
# you should achieve greater than 40% accuracy on
# the training set:

# In[ ]:

model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

solver = Solver(model, data,
    num_epochs=1, batch_size=50,
    update_rule='adam',
    optim_config={
        'learning_rate': 1e-3,
    },
    verbose=True, print_every=20)
solver.train()

# ## Visualize Filters
# You can visualize the first-layer convolutional filters from
# the trained network by running the following:

# In[ ]:

from cs231n.vis_utils import visualize_grid

grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))
plt.imshow(grid.astype('uint8'))
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()

# # Spatial Batch Normalization
# We already saw that batch normalization is a very useful technique
# for training deep fully-connected networks.
# Batch normalization can also be used for convolutional networks,
# but we need to tweak it a bit;
# the modification will be called "spatial batch normalization."
# 
# Normally batch-normalization accepts inputs of shape `(N, D)`
# and produces outputs of shape `(N, D)`,
# where we normalize across the minibatch dimension `N`.
# For data coming from convolutional layers,
# batch normalization needs to accept inputs of shape
# `(N, C, H, W)` and produce outputs of shape `(N, C, H, W)`
# where the `N` dimension gives the minibatch size and the `(H, W)`
# dimensions give the spatial size of the feature map.
# 
# If the feature map was produced using convolutions,
# then we expect the statistics of each feature channel
# to be relatively consistent both between different
# images and different locations within the same image.
# Therefore spatial batch normalization computes a mean and variance
# for each of the `C` feature channels by computing statistics
# over both the minibatch dimension `N`
# and the spatial dimensions `H` and `W`.

# ## Spatial batch normalization: forward
# 
# In the file `cs231n/layers.py`, implement the
# forward pass for spatial batch normalization in the function
# `spatial_batchnorm_forward`. Check your implementation
# by running the following:

# In[ ]:

# Check the training-time forward pass by checking means and variances
# of features both before and after spatial batch normalization

N, C, H, W = 2, 3, 4, 5
x = 4 * np.random.randn(N, C, H, W) + 10

print 'Before spatial batch normalization:'
print '  Shape: ', x.shape
print '  Means: ', x.mean(axis=(0, 2, 3))
print '  Stds: ', x.std(axis=(0, 2, 3))

# Means should be close to zero and stds close to one
gamma, beta = np.ones(C), np.zeros(C)
bn_param = {'mode': 'train'}
out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
print 'After spatial batch normalization:'
print '  Shape: ', out.shape
print '  Means: ', out.mean(axis=(0, 2, 3))
print '  Stds: ', out.std(axis=(0, 2, 3))

# Means should be close to beta and stds close to gamma
gamma, beta = np.asarray([3, 4, 5]), np.asarray([6, 7, 8])
out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
print 'After spatial batch normalization (nontrivial gamma, beta):'
print '  Shape: ', out.shape
print '  Means: ', out.mean(axis=(0, 2, 3))
print '  Stds: ', out.std(axis=(0, 2, 3))

# In[ ]:

# Check the test-time forward pass by running the training-time
# forward pass many times to warm up the running averages, and then
# checking the means and variances of activations after a test-time
# forward pass.

N, C, H, W = 10, 4, 11, 12

bn_param = {'mode': 'train'}
gamma = np.ones(C)
beta = np.zeros(C)
for t in xrange(50):
    x = 2.3 * np.random.randn(N, C, H, W) + 13
    spatial_batchnorm_forward(x, gamma, beta, bn_param)
bn_param['mode'] = 'test'
x = 2.3 * np.random.randn(N, C, H, W) + 13
a_norm, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)

# Means should be close to zero and stds close to one, but will be
# noisier than training-time forward passes.
print 'After spatial batch normalization (test-time):'
print '  means: ', a_norm.mean(axis=(0, 2, 3))
print '  stds: ', a_norm.std(axis=(0, 2, 3))

# ## Spatial batch normalization: backward
# In the file `cs231n/layers.py`, implement the backward pass for spatial batch normalization in the function `spatial_batchnorm_backward`. Run the following to check your implementation using a numeric gradient check:

# In[ ]:

N, C, H, W = 2, 3, 4, 5
x = 5 * np.random.randn(N, C, H, W) + 12
gamma = np.random.randn(C)
beta = np.random.randn(C)
dout = np.random.randn(N, C, H, W)

bn_param = {'mode': 'train'}
fx = lambda x: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]
fg = lambda a: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]
fb = lambda b: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]

dx_num = eval_numerical_gradient_array(fx, x, dout)
da_num = eval_numerical_gradient_array(fg, gamma, dout)
db_num = eval_numerical_gradient_array(fb, beta, dout)

_, cache = spatial_batchnorm_forward(x, gamma, beta, bn_param)
dx, dgamma, dbeta = spatial_batchnorm_backward(dout, cache)
print 'dx error: ', rel_error(dx_num, dx)
print 'dgamma error: ', rel_error(da_num, dgamma)
print 'dbeta error: ', rel_error(db_num, dbeta)


# # Experiment!
# Experiment and try to get the best performance that you can on CIFAR-10 using a ConvNet.
# Here are some ideas to get you started:
# 
# ### Things you should try:
# - Filter size: Above we used 7x7; this makes pretty pictures but smaller filters may be more efficient
# - Number of filters: Above we used 32 filters. Do more or fewer do better?
# - Batch normalization: Try adding spatial batch normalization after convolution layers
# and vanilla batch normalization aafter affine layers. Do your networks train faster?
# - Network architecture: The network above has two layers of trainable parameters.
# Can you do better with a deeper network? You can implement alternative architectures
# in the file `cs231n/classifiers/convnet.py`. Some good architectures to try include:
#     - [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]
#     - [conv-relu-pool]XN - [affine]XM - [softmax or SVM]
#     - [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax or SVM]
# 
# ### Tips for training
# For each network architecture that you try, you should tune the learning rate and
# regularization strength. When doing this there are a couple important things to keep in mind:
# 
# - If the parameters are working well, you should see improvement within a few hundred iterations
# - Remember the course-to-fine approach for hyperparameter tuning:
# start by testing a large range of hyperparameters for just a few training iterations to
#  find the combinations of parameters that are working at all.
# - Once you have found some sets of parameters that seem to work,
# search more finely around these parameters. You may need to train for more epochs.
# 
# ### Going above and beyond
# If you are feeling adventurous there are many other features you can implement to try and
# improve your performance. You are **not required** to implement any of these;
# however they would be good things to try for extra credit.
# 
# - Alternative update steps: For the assignment we implemented SGD+momentum,
# RMSprop, and Adam; you could try alternatives like AdaGrad or AdaDelta.
# - Alternative activation functions such as leaky ReLU, parametric ReLU, or MaxOut.
# - Model ensembles
# - Data augmentation
# 
# If you do decide to implement something extra,
# clearly describe it in the "Extra Credit Description" cell below.
# 
# ### What we expect
# At the very least, you should be able to train a ConvNet that gets at least
# 65% accuracy on the validation set. This is just a lower bound -
# if you are careful it should be possible to get accuracies much higher than that!
# Extra credit points will be awarded for particularly high-scoring models or unique approaches.
# 
# You should use the space below to experiment and train your network.
# The final cell in this notebook should contain the training, validation,
# and test set accuracies for your final trained network.
# In this notebook you should also write an explanation of what you did,
# any additional features that you implemented, and any visualizations or graphs
# that you make in the process of training and evaluating your network.
# 
# Have fun and happy training!

# In[ ]:

# Train a really good model on CIFAR-10


# # Extra Credit Description
# If you implement any additional features for extra credit, clearly describe them here with
# pointers to any code in this or other files if applicable.

"""
Instruction:

In this section, you are asked to train a CNN with different hyperparameters.
To start with training, you need to fill in the incomplete code. There are two
places that you need to complete:
a) Backward pass equations for a convolutional layer.
b) Weight update equations with momentum.

After correctly fill in the code, modify the hyperparameters in "main()".
You can then run this file with the command: "python cnn.py" in your terminal.
The program will automatically check your gradient implementation before start.
The program will print out the training progress, and it will display the
training curve by the end. You can optionally save the model by uncommenting
the lines in "main()". You can also optionally load a trained model by
uncommenting the lines before the training.

Important Notes:
1. The Conv2D function has already been implemented. To implement the backward
pass, you should only need to call the existing Conv2D function, following the
equations from part 2 of the assignment. An efficient solution should be within
10 lines of code.

2. The Conv2D function accepts an optional parameter "pad", by default it will
pad the input with the size of the filter, so that the output has the same
dimension as the input. For example,
-----------------------------------------------------------------------------
Variable                | Shape
-----------------------------------------------------------------------------
x (Inputs)              | [N, H, W, C]
w (Filters)             | [I, J, C, K]
Conv2D(x, w)            | [N, H+I, W+J, C] * [I, J, C, K] = [N, H, W, K]
-----------------------------------------------------------------------------
(N=number of examples, H=height, W=width, C=number of channels,
 I=filter height, J=filter width, K=number of filters)

You can also pass in a tuple (P, Q) to specify the amount you want to pad on
height and width dimension.
For example,
-----------------------------------------------------------------------------
Variable                | Shape
-----------------------------------------------------------------------------
x (Inputs)              | [N, H, W, C]
w (Filters)             | [I, J, C, K]
Conv2D(x, w, pad=(P,Q)) | [N, H+P, W+Q, C] * [I, J, C, K] = [N, H+P-I, W+Q-J, K]
-----------------------------------------------------------------------------

3. It is maybe helpful to use "np.transpose" function to transpose the
dimensions of a variable. For example:
-----------------------------------------------------------------------------
Variable                   | Shape
-----------------------------------------------------------------------------
x (Inputs)                 | [N, H, W, C]
np.transpose(x, [3,1,2,0]) | [C, H, W, N]
-----------------------------------------------------------------------------
"""

from __future__ import division
from __future__ import print_function

from util import LoadData, Load, Save, DisplayPlot
from conv2d import conv2d as Conv2D
from nn import Affine, ReLU, AffineBackward, ReLUBackward, Softmax, CheckGrad, Train, Evaluate

import numpy as np
import matplotlib.pyplot as plt

def InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
            num_outputs):
    """Initializes CNN parameters.

    Args:
        num_channels:  Number of input channels.
        filter_size:   Filter size.
        num_filters_1: Number of filters for the first convolutional layer.
        num_filters_2: Number of filters for the second convolutional layer.
        num_outputs:   Number of output units.

    Returns:
        model:         Randomly initialized network weights.
    """
    W1 = 0.1 * np.random.randn(filter_size, filter_size,
                               num_channels,  num_filters_1)
    W2 = 0.1 * np.random.randn(filter_size, filter_size,
                               num_filters_1, num_filters_2)
    W3 = 0.01 * np.random.randn(num_filters_2 * 64, num_outputs)
    b1 = np.zeros((num_filters_1))
    b2 = np.zeros((num_filters_2))
    b3 = np.zeros((num_outputs))
    model = {
        'W1': W1,
        'W2': W2,
        'W3': W3,
        'b1': b1,
        'b2': b2,
        'b3': b3,
        'W1_vel': W1 * 0,
        'W2_vel': W2 * 0,
        'W3_vel': W3 * 0,
        'b1_vel': b1 * 0,
        'b2_vel': b2 * 0,
        'b3_vel': b3 * 0 
    }
    return model


def MaxPool(x, ratio):
    """Computes non-overlapping max-pooling layer.

    Args:
        x:     Input values.
        ratio: Pooling ratio.

    Returns:
        y:     Output values.
    """
    xs = x.shape
    h = x.reshape([xs[0], int(xs[1] / ratio), ratio,
                   int(xs[2] / ratio), ratio, xs[3]])
    y = np.max(np.max(h, axis=4), axis=2)
    return y


def MaxPoolBackward(grad_y, x, y, ratio):
    """Computes gradients of the max-pooling layer.

    Args:
        grad_y: Gradients wrt. the inputs.
        x:      Input values.
        y:      Output values.

    Returns:
        grad_x: Gradients wrt. the inputs.
    """
    dy = grad_y
    xs = x.shape
    ys = y.shape
    h = x.reshape([xs[0], int(xs[1] / ratio), ratio,
                   int(xs[2] / ratio), ratio, xs[3]])
    y_ = np.expand_dims(np.expand_dims(y, 2), 4)
    dy_ = np.expand_dims(np.expand_dims(dy, 2), 4)
    dy_ = np.tile(dy_, [1, 1, ratio, 1, ratio, 1])
    dx = dy_ * (y_ == h).astype('float')
    dx = dx.reshape([ys[0], ys[1] * ratio, ys[2] * ratio, ys[3]])
    return dx


def Conv2DBackward(grad_y, x, y, w):
    """Computes gradients of the convolutional layer.

    Args:
        grad_y: Gradients wrt. the inputs.
        x:      Input values.
        y:      Output values.

    Returns:
        grad_x: Gradients wrt. the inputs.
        grad_w: Gradients wrt. the weights.
    """
    ###########################
    # Insert your code here.
    # grad_x = ...
    # grad_w = ...
    # return grad_x, grad_w
    ###########################
    
    I = w.shape[0]
    J = w.shape[1]
    x_swapped = np.transpose(x,[3,1,2,0])
    grad_y_swapped = np.transpose(grad_y,[1,2,0,3])
    w_transpose = np.transpose(w[::-1,::-1,:,:],[0,1,3,2])
    grad_w_swapped = Conv2D(x_swapped, grad_y_swapped, pad=(I-1,J-1))

    grad_x = Conv2D(grad_y, w_transpose, pad=(I-1,J-1))
    grad_w = np.transpose(grad_w_swapped,[1,2,0,3])

    return grad_x, grad_w


def CNNForward(model, x):
    """Runs the forward pass.

    Args:
        model: Dictionary of all the weights.
        x:     Input to the network.

    Returns:
        var:   Dictionary of all intermediate variables.
    """
    x = x.reshape([-1, 48, 48, 1])
    h1c = Conv2D(x, model['W1']) + model['b1']
    h1r = ReLU(h1c)
    h1p = MaxPool(h1r, 3)
    h2c = Conv2D(h1p, model['W2']) + model['b2']
    h2r = ReLU(h2c)
    h2p = MaxPool(h2r, 2)
    h2p_ = np.reshape(h2p, [x.shape[0], -1])
    y = Affine(h2p_, model['W3'], model['b3'])
    var = {
        'x': x,
        'h1c': h1c,
        'h1r': h1r,
        'h1p': h1p,
        'h2c': h2c,
        'h2r': h2r,
        'h2p': h2p,
        'h2p_': h2p_,
        'y': y
    }
    return var


def CNNBackward(model, err, var):
    """Runs the backward pass.

    Args:
        model:    Dictionary of all the weights.
        err:      Gradients to the output of the network.
        var:      Intermediate variables from the forward pass.
    """
    dE_dh2p_, dE_dW3, dE_db3 = AffineBackward(err, var['h2p_'], model['W3'])
    dE_dh2p = np.reshape(dE_dh2p_, var['h2p'].shape)
    dE_dh2r = MaxPoolBackward(dE_dh2p, var['h2r'], var['h2p'], 2)
    dE_dh2c = ReLUBackward(dE_dh2r, var['h2c'], var['h2r'])
    dE_dh1p, dE_dW2 = Conv2DBackward(
        dE_dh2c, var['h1p'], var['h2c'], model['W2'])
    dE_db2 = dE_dh2c.sum(axis=2).sum(axis=1).sum(axis=0)
    dE_dh1r = MaxPoolBackward(dE_dh1p, var['h1r'], var['h1p'], 3)
    dE_dh1c = ReLUBackward(dE_dh1r, var['h1c'], var['h1r'])
    _, dE_dW1 = Conv2DBackward(dE_dh1c, var['x'], var['h1c'], model['W1'])
    dE_db1 = dE_dh1c.sum(axis=2).sum(axis=1).sum(axis=0)
    model['dE_dW1'] = dE_dW1
    model['dE_dW2'] = dE_dW2
    model['dE_dW3'] = dE_dW3
    model['dE_db1'] = dE_db1
    model['dE_db2'] = dE_db2
    model['dE_db3'] = dE_db3
    pass


def CNNUpdate(model, eps, momentum):
    """Update NN weights.

    Args:
        model:    Dictionary of all the weights.
        eps:      Learning rate.
        momentum: Momentum.
    """
    ###########################
    # Insert your code here.
    # Update the weights.
    # model['W1'] = ...
    # model['W2'] = ...
    # model['W3'] = ...
    # model['b1'] = ...
    # model['b2'] = ...
    # model['b3'] = ...
    ###########################
    model['W1_vel'] = momentum * model['W1_vel'] + eps * model['dE_dW1']
    model['W2_vel'] = momentum * model['W2_vel'] + eps * model['dE_dW2']
    model['W3_vel'] = momentum * model['W3_vel'] + eps * model['dE_dW3']
    model['b1_vel'] = momentum * model['b1_vel'] + eps * model['dE_db1']
    model['b2_vel'] = momentum * model['b2_vel'] + eps * model['dE_db2']
    model['b3_vel'] = momentum * model['b3_vel'] + eps * model['dE_db3']
    model['W1'] = model['W1'] - model['W1_vel']
    model['W2'] = model['W2'] - model['W2_vel']
    model['W3'] = model['W3'] - model['W3_vel']
    model['b1'] = model['b1'] - model['b1_vel']
    model['b2'] = model['b2'] - model['b2_vel']
    model['b3'] = model['b3'] - model['b3_vel']

def ShowFilters(means, number=0):
  """Show the cluster centers as images."""
  plt.figure(number,figsize=(30,40))
  plt.clf()
  for i in xrange(means.shape[3]):
    plt.subplot(1, means.shape[3], i+1)
    plt.imshow(means[:,:,0, i].reshape(5, 5), cmap=plt.cm.gray)
  plt.draw()
  plt.pause(0.0001)
  plt.savefig('CNNFirstLayerFilters.png')
  raw_input('Press Enter.')

def main(eps, momentum, num_epochs, filter_size, num_filters_1, num_filters_2, batch_size):
    """Trains a CNN."""
    model_fname = 'cnn_model.npz'
    stats_fname = 'cnn_stats.npz'

    # Hyper-parameters. Modify them if needed.
    #eps = 0.1
    #momentum = 0.0
    #num_epochs = 30
    #filter_size = 5
    #num_filters_1 = 8
    #num_filters_2 = 16
    #batch_size = 100

    # Input-output dimensions.
    num_channels = 1
    num_outputs = 7

    # Initialize model.
    model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
                    num_outputs)

    # Uncomment to reload trained model here.
    # model = Load(model_fname)

    # Check gradient implementation.
    print('Checking gradients...')
    x = np.random.rand(10, 48, 48, 1) * 0.1
    CheckGrad(model, CNNForward, CNNBackward, 'W3', x)
    CheckGrad(model, CNNForward, CNNBackward, 'b3', x)
    CheckGrad(model, CNNForward, CNNBackward, 'W2', x)
    CheckGrad(model, CNNForward, CNNBackward, 'b2', x)
    CheckGrad(model, CNNForward, CNNBackward, 'W1', x)
    CheckGrad(model, CNNForward, CNNBackward, 'b1', x)

    # Train model.
    stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps,
                  momentum, num_epochs, batch_size)
    #Uncomment for Q3.4
    #ShowFilters(model['W1'])
    
    # Uncomment if you wish to save the model.
    # Save(model_fname, model)

    # Uncomment if you wish to save the training statistics.
    # Save(stats_fname, stats)

if __name__ == '__main__':
    eps = 0.1
    momentum = 0.0
    num_epochs = 30
    filter_size = 5
    num_filters_1 = 8
    num_filters_2 = 16
    batch_size = 100

    #Q3.1
    #main(eps, momentum, num_epochs, filter_size, num_filters_1, num_filters_2, batch_size)

    #Q3.2
    #main(0.001, momentum, num_epochs, filter_size, num_filters_1, num_filters_2, batch_size)
    #main(0.01, momentum, num_epochs, filter_size, num_filters_1, num_filters_2, batch_size)
    #main(0.1, momentum, num_epochs, filter_size, num_filters_1, num_filters_2, batch_size)
    #main(0.2, momentum, num_epochs, filter_size, num_filters_1, num_filters_2, batch_size)
    #main(1.0, momentum, num_epochs, filter_size, num_filters_1, num_filters_2, batch_size)

    #main(eps, 0.0, num_epochs, filter_size, num_filters_1, num_filters_2, batch_size)
    #main(eps, 0.2, num_epochs, filter_size, num_filters_1, num_filters_2, batch_size)
    #main(eps, 0.45, num_epochs, filter_size, num_filters_1, num_filters_2, batch_size)

    #main(eps, momentum, num_epochs, filter_size, num_filters_1, num_filters_2, 1)
    #main(eps, momentum, num_epochs, filter_size, num_filters_1, num_filters_2, 20)
    #main(eps, momentum, num_epochs, filter_size, num_filters_1, num_filters_2, 40)
    #main(eps, momentum, num_epochs, filter_size, num_filters_1, num_filters_2, 60)
    #main(eps, momentum, num_epochs, filter_size, num_filters_1, num_filters_2, 80)

    #Q3.3
    #main(0.01, 0.9, num_epochs, filter_size, 4, num_filters_2, batch_size)
    #main(0.01, 0.9, num_epochs, filter_size, 8, num_filters_2, batch_size)
    #main(0.01, 0.9, num_epochs, filter_size, 12, num_filters_2, batch_size)
    #main(0.01, 0.9, num_epochs, filter_size, num_filters_1, 4, batch_size)
    #main(0.01, 0.9, num_epochs, filter_size, num_filters_1, 8, batch_size)
    #main(0.01, 0.9, num_epochs, filter_size, num_filters_1, 12, batch_size)

    #Q3.4
    #main(eps, momentum, num_epochs, filter_size, num_filters_1, num_filters_2, batch_size)

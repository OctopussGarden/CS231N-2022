from builtins import range
import numpy as np
from pyparsing import col


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_reshape = np.reshape(x, (x.shape[0], -1))
    out = x_reshape @ w + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = (dout @ w.T).reshape(x.shape)        # shape (N, D) ---> (N, d_1, ... d_k)
    x_reshape = np.reshape(x, (x.shape[0], -1))
    dw = x_reshape.T @ dout
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0., x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * (x>0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # # N = x.shape[0]
    # N =  len(y)

    # scores_exp = np.exp(x - x.max(axis=1, keepdims=True))
    # softmax = scores_exp / \
    #     scores_exp.sum(axis=1, keepdims=True)  # shape (N, C)

    # # loss
    # true_softmax = softmax[range(N), y]  # shape (N,)
    # loss = -np.sum(np.log(true_softmax))
    # loss /= N

    # # gradient
    # softmax[range(N), y] -= 1                  # update P for gradient
    # dx = softmax / N       # calculate gradient

    N = len(y)  # number of samples

    # numerically stable exponents
    P = np.exp(x - x.max(axis=1, keepdims=True))
    # row-wise probabilities (softmax)
    P /= P.sum(axis=1, keepdims=True)

    loss = -np.log(P[range(N), y]).sum() / N     # sum cross entropies as loss

    P[range(N), y] -= 1
    dx = P / N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Compute sample mean and variance
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        std = np.sqrt(var + eps)

        # Normalize the incoming data
        x_hat = (x-mu) / std # x_hat.shape (N, D)
        
        # Scale and shift the normalized data
        out = gamma * x_hat + beta

        # Compute running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
        
        
        # Store cache data
        #####################WHY####################
        shape = bn_param.get("shape", (N, D))
        axis = bn_param.get("axis", 0)
        ############################################
        cache = x, mu, var, std, eps, x_hat, gamma, shape, axis # (N, D) is shape, 0 is axis


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_hat = (x-running_mean) / np.sqrt(running_var + eps) # x_hat.shape (N, D)
        out = gamma * x_hat + beta  # Scale and shift the normalized data

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Formula reference https://blog.csdn.net/litt1e/article/details/105817224

    # Retrieve parameters from cache
    x, mu, var, std, eps, x_hat, gamma, shape, axis = cache
    N = shape[0]
    # Compute gradients
    dx_hat = dout * gamma  # dx_hat.shape (N, D)
    dvar = np.sum(-0.5 * dx_hat * (x-mu) * np.power(var+eps, -1.5), axis=axis)
    dmu = np.sum(- dx_hat / std, axis) + dvar * np.average(-2 * (x-mu), axis)
    dx = dx_hat / std + dvar * 2 * (x-mu) / N + dmu / N
    dgamma = np.sum(dout * x_hat, axis) # x_hat = (x-mu) / std
    dbeta = np.sum(dout, axis)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Ref: 
    # Theory https://cthorey.github.io/blog/2016/backpropagation/
    # Code https://github.com/mantasu/cs231n/blob/master/assignment2/cs231n/layers.py

    _, _, _, std, _, x_hat, gamma, shape, axis = cache
    N = shape[0]

    dgamma = np.sum(dout * x_hat, axis) # x_hat = (x-mu) / std
    dbeta = np.sum(dout, axis)

    dx = dout * gamma / (std * N)
    dx =  N * dx - np.sum(dx, axis) - np.sum(dx * x_hat, axis) * x_hat

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute sample mean and variance
    mu = np.mean(x, axis=1) # shape (N,)
    var = np.var(x, axis=1) # shape (N,)
    std = np.sqrt(var + eps) # shape (N,)

    # Normalize the incoming data
    x_hat = (x-mu[:, np.newaxis]) / std[:, np.newaxis]  # x_hat.shape (N, D)
    
    # Scale and shift the normalized data
    out = gamma * x_hat + beta
    
    # Store cache data
    shape = ln_param.get("shape", x.shape)
    axis = ln_param.get("axis", 1)
    
    cache = x, mu, var, std, eps, x_hat, gamma, shape, axis # (N, D) is shape, 0 is axis


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    _, _, _, std, _, x_hat, gamma, shape, axis = cache
    D = shape[1] # This parameter should be shape[axis]

    dgamma = np.sum(dout * x_hat, 0)  # x_hat = (x-mu) / std
    dbeta = np.sum(dout, 0)

    dx = dout * gamma / (std[:, np.newaxis] * D)
    dx = D * dx - np.sum(dx, axis)[:, np.newaxis] - np.sum(dx * x_hat, axis)[:, np.newaxis] * x_hat

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.uniform(0., 1., x.shape) <= p).astype(np.float32)
        out = np.multiply(x, mask)
        out = out / p # make drop layers' mean back to normal
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = np.multiply(dout, mask) / dropout_param['p']

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # get shapes
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    pad, stride  = conv_param['pad'], conv_param['stride']


    H_prime = int(1 + (H + 2 * pad - HH) / stride)
    W_prime = int(1 + (W + 2 * pad - WW) / stride)

    # # Reshape w
    # w_reshaped = np.reshape(w, (N, C, -1))
    # pad x
    pad_tuple = ((0,0), (0,0), (pad,pad), (pad,pad)) # only pad last two dimensions
    x_pad = np.pad(x, pad_width=pad_tuple, mode='constant', constant_values=0.)

    # for n in range(N):
    #   for c in range(C):
    #     x_pad = np.pad(x[n,c,:,:], pad_width=pad, mode='constant', constant_values=0.)
    
    # Compute output

    out = np.zeros((N, F, H_prime, W_prime))

    for n in range(N):
      for f in range(F):
        w_slice = w[f,:,:,:]
        for i in range(H_prime):
          start_row = stride * i
          for j in range(W_prime):
            start_col = stride * j
            x_slice = x_pad[n, :, start_row:start_row+HH, start_col:start_col+WW] # Receptive Field --- shape(N, C, HH, WW)
            out[n, f, i, j] = np.sum(np.multiply(w_slice, x_slice)) + b[f]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # NOTICE: dout.shape is (N, F, H_prime, W_prime)

    # Retrieve parameters from cache.
    x, w, b, conv_param = cache
    pad, stride  = conv_param['pad'], conv_param['stride']
    # Get shapes
    N, _, _, _ = x.shape
    F, _, HH, WW = w.shape
    _, _, H_prime, W_prime = dout.shape

    # get x_pad
    pad_tuple = ((0,0), (0,0), (pad,pad), (pad,pad))
    x_pad = np.pad(x, pad_width=pad_tuple, mode='constant', constant_values=0.)

    # Initialize gradients as numpy array
    dx, dw, db = np.zeros_like(x_pad), np.zeros_like(w), np.zeros_like(b)

    # Compute gradients

    # dx
    for n in range(N):
      for f in range(F):
        for i in range(H_prime):
          start_row = stride * i
          for j in range(W_prime):
            start_col = stride * j
            dout_slice = dout[n, f, i, j] # scalar value
            w_slice = w[f,:,:,:] # (1, C, HH, WW)
            dx[n, :, start_row:start_row+HH, start_col:start_col+WW] += np.multiply(dout_slice, w_slice) # (1, C, HH, WW)
    
    dx = dx[:,:,pad:-pad,pad:-pad]  # (1, C, H, W)

    # dw
    for n in range(N):
      for f in range(F):
        for i in range(H_prime):
          start_row = stride * i
          for j in range(W_prime):
            start_col = stride * j
            x_slice = x_pad[n, :, start_row:start_row+HH, start_col:start_col+WW] # (1, C, HH, WW)
            dout_slice = dout[n, f, i, j]
            dw[f, :, :, :] += np.multiply(dout_slice, x_slice)

    # db
    db = np.sum(dout, axis=(0, 2, 3)) # db.shape (F,)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Retrieve parameters
    N, C, H, W = x.shape
    ph, pw, s = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    
    # Calculate H_prime and W_prime
    H_prime = int(1 + (H - ph) / s)
    W_prime = int(1 + (W - pw) / s)

    # Calculate output
    ## Initialization
    out = np.zeros((N, C, H_prime, W_prime))
    ## Calculation
    for n in range(N):
      for c in range(C):
        for i in range(H_prime):
          start_row = s * i
          for j in range(W_prime):
            start_col = s * j
            out[n, c, i, j] = np.max(x[n, c, start_row:start_row+ph, start_col:start_col+pw])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Retrieve parameters
    x, pool_param = cache
    N, C, H, W = x.shape
    ph, pw, s = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    

    # Calculate H_prime and W_prime
    H_prime = int(1 + (H - ph) / s)
    W_prime = int(1 + (W - pw) / s)

    # Calculate deriatives
    ## Initialization
    dx = np.zeros_like(x)
    ## Calculation, dx update its argmax entry only.
    for n in range(N):
      for c in range(C):
        for i in range(H_prime):
          start_row = s * i
          for j in range(W_prime):
            start_col = s * j
            x_slice = x[n, c, start_row:start_row+ph, start_col:start_col+pw] # (ph, pw)
            index = np.argmax(x_slice)
            row_idx, col_idx = np.unravel_index(index, (ph, pw)) # get index of the maximum entry
            dx[n, c, start_row:start_row+ph, start_col:start_col+pw][row_idx, col_idx] += dout[n, c, i, j] #scalar

    # print(dx)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    x_reshaped = (np.transpose(x, (0,3,2,1))).reshape((-1, C)) # shape (N*H*W, C), convinent to call vanilla batch normalization forward
    out, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param) # shape (N*H*W, C)
    out = np.transpose((out.reshape(N, W, H, C)), (0,3,2,1)) # reshape out to (N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = dout.shape
    dout_reshaped = (np.transpose(dout, (0, 3, 2, 1))).reshape((-1, C)) # shape (N*H*W, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout_reshaped, cache)
    dx = np.transpose((dx.reshape(N, W, H, C)), (0,3,2,1)) # reshape to (N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #### Solution 2 - reshape to (N*G, H*W*C/G)
    N, _, _, _ = x.shape

    # Reshape input x
    x_reshaped = np.reshape(x, (N*G, -1)) # (N*G, H*W*C/G)

    # Calculation statistics
    mu = np.mean(x_reshaped, axis=1) # (N*G,)
    var = np.var(x_reshaped, axis=1)  
    std = np.sqrt(var + eps) # (N*G,)

    # Normalize the incoming data
    x_hat = (x_reshaped-mu[:, np.newaxis]) / std[:, np.newaxis] # (N*G, H*W*C/G)

    # Reshape normalized data back to 
    x_hat = np.reshape(x_hat, x.shape) # (N, C, H, W)

    # Scale and shift the normalized data
    out = gamma * x_hat + beta # (N, C, H, W)

    # Store cache data
    shape = gn_param.get("shape", x_reshaped.shape)
    axis = gn_param.get("axis", 1)

    cache = G, std, x_hat, gamma, shape, axis



    ######### Solution 1 - hyperparameters have loop index issues

    # N, C, H, W = x.shape

    # # Reshape input x
    # x_reshaped = (np.transpose(x, (0,3,2,1))).reshape((-1, C)) # shape (N*H*W, C)

    # # Initialization
    # out = np.zeros_like(x_reshaped)

    # # Get group size
    # group_size = C // G

    # # create variable for cache storage
    # std, x_hat = [None]*group_size, [None]*group_size

    # for g in range(G):
    #   # Get block
    #   x_block = x_reshaped[:, g*group_size:(g+1)*group_size]

    #   # Compute sample mean and variance
    #   mu = np.mean(x_block, axis=1) # shape (N*H*W,)
    #   var = np.var(x_block, axis=1) # shape (N*H*W,)
    #   std[g] = np.sqrt(var + eps)      # shape (N*H*W,)

    #   # Normalize the incoming data
    #   x_hat[g] = (x_block-mu[:, np.newaxis]) / std[g][:, np.newaxis]  # x_hat.shape (N*H*W, C)
      
    #   #####################TODO: Here should ne C related loop variable, not g
    #   #####################Otherwise gamma and beta's shape would be a problem
    #   # Scale and shift the normalized data
    #   out[:, g*group_size:(g+1)*group_size] = gamma[:,g,:,:] * x_hat[g] + beta[:,g,:,:]
      
    # # Store cache data
    # shape = gn_param.get("shape", x_reshaped.shape)
    # axis = gn_param.get("axis", 1)
    
    # cache = G, group_size, std, x_hat, gamma, shape, axis # (N*H*W, C) is shape, 0 is axis

    # # Reshape out to (N, C, H, W)
    # out = np.transpose((out.reshape(N, W, H, C)), (0,3,2,1))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #### Solution 2 - reshape to (N*G, H*W*C/G)
    N, C, H, W = dout.shape
    G, std, x_hat, gamma, shape, axis = cache
    D = shape[axis] # H*W*C/G

    # # Reshape dout
    x_hat_reshaped = np.reshape(x_hat, (N*G, -1))  # (N*G, H*W*C/G)
    # dout_reshaped = np.reshape(N*G, -1) # (N*G, H*W*C/G)

    # Calculation
    dgamma = np.sum(dout * x_hat, axis=(0,2,3))  # (C,)
    dbeta = np.sum(dout, axis=(0,2,3)) # (C,)

    dgamma = np.reshape(dgamma, gamma.shape) # (1, C, 1, 1)
    dbeta = np.reshape(dbeta, gamma.shape) # (1, C, 1, 1)

    # Preprocess dx to simplify calculation and reshape it to 2-dim array
    dx = dout * gamma
    dx = np.reshape(dx, shape) # (N*G, H*W*C/G)
    dx /= (std[:, np.newaxis] * D) # (N*G, H*W*C/G)

    dx = D * dx - np.sum(dx, axis)[:, np.newaxis] - \
        np.sum(dx * x_hat_reshaped, axis)[:, np.newaxis] * x_hat_reshaped # (N*G, H*W*C/G)
    
    dx = np.reshape(dx, dout.shape) # (N, C, H, W)




    ######### Solution 1 - hyperparameters have loop index issues
    # # Retrieve parameters
    # N, C, H, W = dout.shape
    # G, group_size, std, x_hat, gamma, shape, axis = cache

    # # Reshape dout
    # dout_reshaped = (np.transpose(dout, (0, 3, 2, 1))).reshape((-1, C)) # shape (N*H*W, C)

    # # Initialization
    # dx = np.zeros_like(dout_reshaped)
    # dgamma, dbeta = [None]*C, [None]*C  # (1, C, 1, 1)

    # for g in range(G):
    #   # Get block
    #   dout_block = dout_reshaped[:, g*group_size:(g+1)*group_size]

    #   # Calculation
    #   dgamma[g*group_size:(g+1)*group_size] = np.sum(dout_block * x_hat[g], 0)  # x_hat = (x-mu) / std
    #   dbeta[g*group_size:(g+1)*group_size] = np.sum(dout_block, 0)

    #   dx[:, g*group_size:(g+1)*group_size] = dout_block * gamma[:, g *
    #                                                             group_size:(g+1)*group_size, :, :] / (std[g][:, np.newaxis] * C)
    #   dx[:, g*group_size:(g+1)*group_size] = C * dx[:, g*group_size:(g+1)*group_size] - np.sum(dx[:, g*group_size:(g+1)*group_size], axis)[:, np.newaxis] - \
    #       np.sum(dx[:, g*group_size:(g+1)*group_size] * x_hat[g], axis)[:, np.newaxis] * x_hat[g]
    
    # # Reshape
    # dx = np.transpose((dx.reshape(N, W, H, C)), (0,3,2,1))
    # dgamma = np.reshape(np.array(dgamma), (1, C, 1, 1))
    # dbeta = np.reshape(np.array(dbeta), (1, C, 1, 1))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta

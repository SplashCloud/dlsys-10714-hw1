"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filename) as f:
      magic_number = f.read(4)
      N_images = struct.unpack(">i", f.read(4))[0]
      row = struct.unpack(">i", f.read(4))[0]
      col = struct.unpack(">i", f.read(4))[0]
      print(f'N_images={N_images}, row={row}, col={col}')
      X = []
      for _ in range(N_images):
        item = []
        for _ in range(row):
          for _ in range(col):
            pixel = struct.unpack(">B", f.read(1))[0]
            item.append(pixel)
        X.append(item)
    
    with gzip.open(label_filename) as f:
      magic_number = f.read(4)
      N_labels = struct.unpack(">i", f.read(4))[0]
      print(f'N_labels={N_labels}')
      y = []
      for _ in range(N_labels):
        label = struct.unpack(">B", f.read(1))[0]
        y.append(label)
    
    def norm(arr):
      min_v = np.min(arr)
      max_v = np.max(arr)
      return (arr - min_v) / (max_v - min_v)
    
    X = np.array(X, dtype=np.float32) 
    return norm(X), np.array(y, dtype=np.uint8)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    return (ndl.log(ndl.exp(Z).sum(axes=(1,))) - (Z * y_one_hot).sum(axes=(1,))).sum(axes=(0,)) / Z.shape[0]
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    n = X.shape[0]
    total_batch = []
    batch_num = n // batch
    i = 1
    while i <= batch_num:
      total_batch.append(
        (
          X[(i-1)*batch : i*batch],
          y[(i-1)*batch : i*batch]
        )
      )
      i += 1
    if n % batch != 0:
      total_batch.append(
        (
          X[(i-1)*batch : ],
          y[(i-1)*batch : ]
        )
      )

    cnt = 0
    for single_batch in total_batch:
      X0, y0 = single_batch
      X_batch = ndl.Tensor(X0) # B x n
      y_one_hot = np.zeros(shape=(X_batch.shape[0], W2.shape[1]))
      y_one_hot[np.arange(y_one_hot.shape[0]), y0] = 1
      y_batch = ndl.Tensor(y_one_hot) # B x k
      # W1: n x d, W2: d x k
      Z = ndl.matmul(ndl.relu(ndl.matmul(X_batch, W1)), W2) # B x k
      loss = softmax_loss(Z, y_batch)
      loss.backward()
      loss.detach()
      cnt += 1
      print(f"finish backward: {cnt}")
      # W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
      # W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
      W1 = (W1 - lr * W1.grad).detach()
      W2 = (W2 - lr * W2.grad).detach()
      ######## ERROR UPDATE #########
      # W1 = W1 - lr * W1.grad #
      # W2 = W2 - lr * W2.grad #
      # W1 and W2 still point to origin computation graph, so that it will eat up the memory
      ##################################
    print(f"totoal backward: {cnt}")
    return (W1, W2)
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)

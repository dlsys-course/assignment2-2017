from dlsys import autodiff as ad
from dlsys import ndarray, gpu_op
import numpy as np

import argparse
import six.moves.cPickle as pickle
import gzip
import os


def load_mnist_data(dataset):
    """ Load the dataset
    Code adapted from http://deeplearning.net/tutorial/code/logistic_sgd.py

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    """
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('Loading data...')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix), np.float32
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector), np.int64 that has the same length
    # as the number of rows in the input. It should give the target
    # to the example with the same index in the input.
    return train_set, valid_set, test_set


def convert_to_one_hot(vals):
    """Helper method to convert label array to one-hot array."""
    one_hot_vals = np.zeros((vals.size, vals.max()+1))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals


def sgd_update_gpu(param, grad_param, learning_rate):
    """Helper GPU SGD update method. Avoids copying NDArray to cpu."""
    assert isinstance(param, ndarray.NDArray)
    assert isinstance(grad_param, ndarray.NDArray)
    gpu_op.matrix_elementwise_multiply_by_const(
        grad_param, -learning_rate, grad_param)
    gpu_op.matrix_elementwise_add(param, grad_param, param)


def mnist_logreg(executor_ctx=None, num_epochs=10, print_loss_val_each_epoch=False):
    print("Build logistic regression model...")

    W1 = ad.Variable(name="W1")
    b1 = ad.Variable(name="b1")
    X = ad.Variable(name="X")
    y_ = ad.Variable(name="y_")

    z1 = ad.matmul_op(X, W1)
    y = z1 + ad.broadcastto_op(b1, z1)

    loss = ad.softmaxcrossentropy_op(y, y_)

    grad_W1, grad_b1 = ad.gradients(loss, [W1, b1])
    executor = ad.Executor([loss, grad_W1, grad_b1, y], ctx=executor_ctx)

    # Read input data
    datasets = load_mnist_data("mnist.pkl.gz")
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Set up minibatch
    batch_size = 1000
    n_train_batches = train_set_x.shape[0] // batch_size
    n_valid_batches = valid_set_x.shape[0] // batch_size

    print("Start training loop...")

    # Initialize parameters
    W1_val = np.zeros((784, 10))
    b1_val = np.zeros((10))
    X_val = np.empty(shape=(batch_size, 784), dtype=np.float32)
    y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)
    valid_X_val = np.empty(shape=(batch_size, 784), dtype=np.float32)
    valid_y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)
    if ndarray.is_gpu_ctx(executor_ctx):
        W1_val = ndarray.array(W1_val, ctx=executor_ctx)
        b1_val = ndarray.array(b1_val, ctx=executor_ctx)
        X_val = ndarray.array(X_val, ctx=executor_ctx)
        y_val = ndarray.array(y_val, ctx=executor_ctx)

    lr = 1e-3
    for i in range(num_epochs):
        print("epoch %d" % i)
        for minibatch_index in range(n_train_batches):
            minibatch_start = minibatch_index * batch_size
            minibatch_end = (minibatch_index + 1) * batch_size
            X_val[:] = train_set_x[minibatch_start:minibatch_end]
            y_val[:] = convert_to_one_hot(
                train_set_y[minibatch_start:minibatch_end])
            loss_val, grad_W1_val, grad_b1_val, _ = executor.run(
                feed_dict = {X: X_val, y_: y_val, W1: W1_val, b1: b1_val})
            # SGD update
            if (executor_ctx is None):
                W1_val = W1_val - lr * grad_W1_val
                b1_val = b1_val - lr * grad_b1_val
            else:
                sgd_update_gpu(W1_val, grad_W1_val, lr)
                sgd_update_gpu(b1_val, grad_b1_val, lr)
        if print_loss_val_each_epoch:
            if isinstance(loss_val, ndarray.NDArray):
                print(loss_val.asnumpy())
            else:
                print(loss_val)

    correct_predictions = []
    for minibatch_index in range(n_valid_batches):
        minibatch_start = minibatch_index * batch_size
        minibatch_end = (minibatch_index + 1) * batch_size
        valid_X_val[:] = valid_set_x[minibatch_start:minibatch_end]
        valid_y_val[:] = convert_to_one_hot(
            valid_set_y[minibatch_start:minibatch_end])
        _, _, _, valid_y_predicted = executor.run(
            feed_dict={
                        X: valid_X_val,
                        y_: valid_y_val,
                        W1: W1_val,
                        b1: b1_val},
            convert_to_numpy_ret_vals=True)
        correct_prediction = np.equal(
            np.argmax(valid_y_val, 1),
            np.argmax(valid_y_predicted, 1)).astype(np.float)
        correct_predictions.extend(correct_prediction)
    accuracy = np.mean(correct_predictions)
    # validation set accuracy=0.928200
    print("validation set accuracy=%f" % accuracy)


def mnist_mlp(executor_ctx=None, num_epochs=10, print_loss_val_each_epoch=False):
    print("Build 3-layer MLP model...")

    W1 = ad.Variable(name="W1")
    W2 = ad.Variable(name="W2")
    W3 = ad.Variable(name="W3")
    b1 = ad.Variable(name="b1")
    b2 = ad.Variable(name="b2")
    b3 = ad.Variable(name="b3")
    X = ad.Variable(name="X")
    y_ = ad.Variable(name="y_")

    # relu(X W1+b1)
    z1 = ad.matmul_op(X, W1)
    z2 = z1 + ad.broadcastto_op(b1, z1)
    z3 = ad.relu_op(z2)

    # relu(z3 W2+b2)
    z4 = ad.matmul_op(z3, W2)
    z5 = z4 + ad.broadcastto_op(b2, z4)
    z6 = ad.relu_op(z5)

    # softmax(z5 W2+b2)
    z7 = ad.matmul_op(z6, W3)
    y = z7 + ad.broadcastto_op(b3, z7)

    loss = ad.softmaxcrossentropy_op(y, y_)

    grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3 = ad.gradients(
        loss, [W1, W2, W3, b1, b2, b3])
    executor = ad.Executor(
        [loss, grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3, y],
        ctx=executor_ctx)

    # Read input data
    datasets = load_mnist_data("mnist.pkl.gz")
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    # Set up minibatch
    batch_size = 1000
    n_train_batches = train_set_x.shape[0] // batch_size
    n_valid_batches = valid_set_x.shape[0] // batch_size

    print("Start training loop...")

    # Initialize parameters
    rand = np.random.RandomState(seed=123)
    W1_val = rand.normal(scale=0.1, size=(784, 256))
    W2_val = rand.normal(scale=0.1, size=(256, 100))
    W3_val = rand.normal(scale=0.1, size=(100, 10))
    b1_val = rand.normal(scale=0.1, size=(256))
    b2_val = rand.normal(scale=0.1, size=(100))
    b3_val = rand.normal(scale=0.1, size=(10))
    X_val = np.empty(shape=(batch_size, 784), dtype=np.float32)
    y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)
    valid_X_val = np.empty(shape=(batch_size, 784), dtype=np.float32)
    valid_y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)
    if ndarray.is_gpu_ctx(executor_ctx):
        W1_val = ndarray.array(W1_val, ctx=executor_ctx)
        W2_val = ndarray.array(W2_val, ctx=executor_ctx)
        W3_val = ndarray.array(W3_val, ctx=executor_ctx)
        b1_val = ndarray.array(b1_val, ctx=executor_ctx)
        b2_val = ndarray.array(b2_val, ctx=executor_ctx)
        b3_val = ndarray.array(b3_val, ctx=executor_ctx)
        X_val = ndarray.array(X_val, ctx=executor_ctx)
        y_val = ndarray.array(y_val, ctx=executor_ctx)

    lr = 1.0e-3
    for i in range(num_epochs):
        print("epoch %d" % i)
        for minibatch_index in range(n_train_batches):
            minibatch_start = minibatch_index * batch_size
            minibatch_end = (minibatch_index + 1) * batch_size
            X_val[:] = train_set_x[minibatch_start:minibatch_end]
            y_val[:] = convert_to_one_hot(
                train_set_y[minibatch_start:minibatch_end])
            loss_val, grad_W1_val, grad_W2_val, grad_W3_val, \
                grad_b1_val, grad_b2_val, grad_b3_val, _ = executor.run(
                    feed_dict={
                        X: X_val,
                        y_: y_val,
                        W1: W1_val,
                        W2: W2_val,
                        W3: W3_val,
                        b1: b1_val,
                        b2: b2_val,
                        b3: b3_val})
            # SGD update
            if (executor_ctx is None):
                W1_val = W1_val - lr * grad_W1_val
                W2_val = W2_val - lr * grad_W2_val
                W3_val = W3_val - lr * grad_W3_val
                b1_val = b1_val - lr * grad_b1_val
                b2_val = b2_val - lr * grad_b2_val
                b3_val = b3_val - lr * grad_b3_val
            else:
                sgd_update_gpu(W1_val, grad_W1_val, lr)
                sgd_update_gpu(W2_val, grad_W2_val, lr)
                sgd_update_gpu(W3_val, grad_W3_val, lr)
                sgd_update_gpu(b1_val, grad_b1_val, lr)
                sgd_update_gpu(b2_val, grad_b2_val, lr)
                sgd_update_gpu(b3_val, grad_b3_val, lr)
        if print_loss_val_each_epoch:
            if isinstance(loss_val, ndarray.NDArray):
                print(loss_val.asnumpy())
            else:
                print(loss_val)

    correct_predictions = []
    for minibatch_index in range(n_valid_batches):
        minibatch_start = minibatch_index * batch_size
        minibatch_end = (minibatch_index + 1) * batch_size
        valid_X_val[:] = valid_set_x[minibatch_start:minibatch_end]
        valid_y_val[:] = convert_to_one_hot(
            valid_set_y[minibatch_start:minibatch_end])
        _, _, _, _, _, _, _, valid_y_predicted = executor.run(
            feed_dict={
                X: valid_X_val,
                y_: valid_y_val,
                W1: W1_val,
                W2: W2_val,
                W3: W3_val,
                b1: b1_val,
                b2: b2_val,
                b3: b3_val},
            convert_to_numpy_ret_vals=True)
        correct_prediction = np.equal(
            np.argmax(valid_y_val, 1),
            np.argmax(valid_y_predicted, 1)).astype(np.float)
        correct_predictions.extend(correct_prediction)
    accuracy = np.mean(correct_predictions)
    # validation set accuracy=0.970800
    print("validation set accuracy=%f" % accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        help="Choose model: all, logreg, mlp", default="all")
    parser.add_argument(
        "-c", "--executor_context",
        help="Choose executor context: numpy, gpu", default="numpy")
    parser.add_argument(
        "-e", "--num_epoch",
        help="Provide number of epochs to train.", type=int, default=10)
    parser.add_argument(
        "-l", "--print_loss_val_each_epoch",
        help="Print loss value at the end of each epoch", action="store_true")
    args = parser.parse_args()

    models = []
    executor_ctx = None
    print_loss_val_each_epoch = False
    if args.model == "logreg":
        models = [mnist_logreg]
    elif args.model == "mlp":
        models = [mnist_mlp]
    elif args.model == "all":
        models = [mnist_logreg, mnist_mlp]

    if args.executor_context == "numpy":
        executor_ctx = None
    elif args.executor_context == "gpu":
        # Assume only use gpu 0.
        executor_ctx = ndarray.gpu(0)

    if args.print_loss_val_each_epoch:
        print_loss_val_each_epoch = True

    num_epochs = args.num_epoch
    for m in models:
        m(executor_ctx, num_epochs, print_loss_val_each_epoch)

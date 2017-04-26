export PYTHONPATH="${PYTHONPATH}:/path/to/gpu_executor/python"

python tests/test_array.py

Use Python ctypes to expose C API
Implement simple GPU kernels for DL ops.

# Assignment 2: GPU Graph Executor

In this assignment, we would implement a GPU graph executor that can train simple neural nets such as multilayer perceptron models.

Our code should be able to construct a simple MLP model using computation graph API implemented in Assignment 1, and train and test the model using either numpy or GPU.

Key concepts and data structures that we would need to implement are
- Shape inference on computation graph given input shapes.
- GPU executor memory management for computation graph.
- GPU kernel implementations of common kernels, e.g. Relu, MatMul, Softmax.

## Overview of Module API and Data Structures

### Special notes here:

## What you need to do?
Understand the code skeleton and tests. Fill in implementation wherever marked """TODO: Your code here""".
There are only files with TODOs for you.
- python/dlsys/autodiff.py
- src/gpu_op.cu

## Tests cases
We have 12 tests in tests/test_gpu_op.py. We would grade your GPU kernel implementations based on those tests.

Run all tests with
```bash
# sudo pip install nose
nosetests -v tests/test_gpu_op.py
```

Run neural nets training and testing with
```bash
# see cmd options with 
# python tests/mnist_dlsys.py -h

# run logistic regression on numpy
python tests/mnist_dlsys.py -l -m logreg -c numpy
# run logistic regression on gpu
python tests/mnist_dlsys.py -l -m logreg -c gpu
# run MLP on numpy
python tests/mnist_dlsys.py -l -m mlp -c numpy
# run MLP on gpu
python tests/mnist_dlsys.py -l -m mlp -c gpu

```

If your implementation is correct, you would see
- generally decreasing loss value with epochs
- your dev set accuracy for logreg about 92% and MLP about 97% for mnist

Profile GPU execution with
```bash
nvprof python tests/mnist_dlsys.py -l -m mlp -c gpu
```

If GPU memory management is done right, e.g. reuse GPU memory across each executor.run, your cudaMalloc "Calls" should not increase with number of training epochs.
```bash
# Run 10 epochs
nvprof python tests/mnist_dlsys.py -l -m mlp -c gpu -e 10
#==2263== API calls:
#Time(%)      Time     Calls       Avg       Min       Max  Name
# 10.19%  218.65ms        64  3.4164ms  8.5130us  213.90ms  cudaMalloc

# Run 30 epochs
nvprof python tests/mnist_dlsys.py -l -m mlp -c gpu -e 30
#==4333== API calls:
#Time(%)      Time     Calls       Avg       Min       Max  Name
#  5.80%  340.74ms        64  5.3240ms  15.877us  333.80ms  cudaMalloc
```


### Bonus points

### Grading rubrics
- test_gpu_op.test_array_set ... 1 pt
- test_gpu_op.test_broadcast_to ... 1 pt
- test_gpu_op.test_reduce_sum_axis_zero ... 1 pt
- test_gpu_op.test_matrix_elementwise_add ... 1 pt
- test_gpu_op.test_matrix_elementwise_add_by_const ... 1 pt
- test_gpu_op.test_matrix_elementwise_multiply ... 1 pt
- test_gpu_op.test_matrix_elementwise_multiply_by_const ... 1 pt
- test_gpu_op.test_matrix_multiply ... 4 pt
- test_gpu_op.test_relu ... 1 pt
- test_gpu_op.test_relu_gradient ... 1 pt
- test_gpu_op.test_softmax ... 2 pt
- test_gpu_op.test_softmax_cross_entropy ... Implemented by us.

- bonus (?) ... ? pt

## Submitting your work

Please submit your autodiff.tar.gz to Catalyst dropbox under [Assignment 2](https://catalyst.uw.edu/collectit/assignment/arvindk/40126/159878).
```bash
# compress
tar czvf gpu_executor.tar.gz gpu_executor/
```

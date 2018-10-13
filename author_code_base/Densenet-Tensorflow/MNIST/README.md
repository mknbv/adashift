# Densenet on MNIST

## Requirements

* Python 3.5 
* tf-1.9

##Getting started

###Preparing data
Download MNIST dataset and extract to directory ``` ./MNIST_data```.
```shell
./MNIST_data
    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz
    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
```

###Training

```shell
python Densenet_MNIST.py
```
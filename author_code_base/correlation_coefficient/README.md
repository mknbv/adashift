# Correlation coefficient

## Requirements

* Python 3.5
* TF-1.9

##Getting started

### Preparing data
If you want to generate gradients by training a multilayer perceptron on MNIST, 
then you need to prepare the MNIST dataset.

Download MNIST dataset and extract to directory ``` ./MNIST_data```.
```shell
./MNIST_data
    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz
    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
```

### Running

```shell
python main.py --exp_name cors
``` 

Results of correlation coefficient will be printed on terminal.
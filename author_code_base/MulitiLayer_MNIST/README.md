# Multilayer Perceptron on MNIST

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

####run SGD without momentum.
```shell
python neural_network_raw.py \
	--optimizer_name sgd --learning_rate 0.001 
```

####run Adam without momentum.
```shell
python neural_network_raw.py \
	--optimizer_name adam --learning_rate 0.001 \
	--beta1 0.0 --beta2 0.999   
```
####run AMSGrad without momentum.
```shell
python neural_network_raw.py \
	--optimizer_name amsgrad --learning_rate 0.001 \
	--beta1 0.0 --beta2 0.999   
```

####run AdaShift with max operation and without momentum.
```shell
python neural_network_raw.py \
	--optimizer_name adashift --learning_rate 0.01 \
	--beta1 0.0 --beta2 0.999  --pred_g_op max --keep_num 10
```

####run AdaShift without specific operation and without momentum.
```shell
python neural_network_raw.py \
	--optimizer_name adashift --learning_rate 0.001 \
	--beta1 0.0 --beta2 0.999  --pred_g_op none --keep_num 10
```

### Training Result
All datas of result are stored in directory ```"./log "```.
Every sub directory is named as:
```"run time ID"+"_"+"optimizer name"+"_"+"operation on previous gradients"+"_"+"average window size"+"_"+"learning rate"+"_"+"beta1"+"_"+"beta2"```.
You can obtain the raw ```".npy"``` data from ```"sub directory/result_data"```
And you can write you own script to plot the resutl.

### PLot
The script ```draw_TF_result.py``` is a reference script to draw the plot where you need to change the "```plot_target```" list in the script.

```shell
python draw_TF_result
```


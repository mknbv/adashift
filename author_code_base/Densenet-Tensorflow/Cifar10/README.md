# Densenet on Cifar10

##Requirements

* Python 3.5
* tf-1.9

##Getting started

###Preparing data
Download Cifar10 dataset and extract to directory ``` ./cifar-10-batches-py```.
```shell
cifar-10-batches-py
├── batches.meta
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── readme.html
└── test_batch
```

###Training
####run SGD.
```shell
python Densenet_Cifar10.py \
	--optimizer_name sgd --init_learning_rate 0.001 
```

####run Adam.
```shell
python Densenet_Cifar10.py \
	--optimizer_name adam --init_learning_rate 0.001 \
	--beta1 0.9 --beta2 0.999   
```
####run AMSGrad.
```shell
python Densenet_Cifar10.py \
	--optimizer_name amsgrad --init_learning_rate 0.001 \
	--beta1 0.9 --beta2 0.999   
```

####run AdaShift with max operation.
```shell
python Densenet_Cifar10.py \
	--optimizer_name adashift --init_learning_rate 0.01 \
	--beta1 0.9 --beta2 0.999  --pred_g_op max --keep_num 10
```
### Training Result
All datas of result are stored in directory ```"./log "```.
Every sub directory is named as:
```"run time ID"+"_"+"optimizer name"+"_"+"operation on previous gradients"+"_"+"average window size"+"_"+"learning rate"+"_"+"beta1"+"_"+"beta2"```.
You can obtain the raw ```".npy"``` data from ```"sub directory/result_data"```

Meanwhile, you can use tensorboard to visualize the reuslts with logdir as ```"logs"```.
```shell
tensorboard --logdir logs
```

# Resnet on Cifar10

##Requirements

* Python 3.5
* tf-1.9
* keras 2.2.2

##Getting started
###Training
####run Adam.
```shell
python main.py \
	--optimizer_name adam --lr 0.001 \
	--beta1 0.9 --beta2 0.999   
```
####run AMSGrad.
```shell
python main.py \
	--optimizer_name amsgrad --lr 0.001 \
	--beta1 0.9 --beta2 0.999   
```

####run AdaShift with max operation.
```shell
python main.py \
	--optimizer_name adashift --lr 0.01 \
	--beta1 0.9 --beta2 0.999  --pred_g_op max --keep_num 10
```
### Training Result
All datas of result are stored in directory ```"./log "```.
Every sub directory is named as:
```"run time ID"+"_"+"optimizer name"+"_"+"operation on previous gradients"+"_"+"average window size"+"_"+"learning rate"+"_"+"beta1"+"_"+"beta2"```.

You can use tensorboard to visualize the reuslts with logdir as ```"logs"```.
```shell
tensorboard --logdir logs
```

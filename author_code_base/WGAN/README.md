# WGAN-GP

##Requirements
* Python 3.5
* tf-1.5

##Training
In this experiment, we set beta1 = 0, beta2 = 0.999 and operation on previous gradient is ```max```. Then, the best learning rate for each optimizer is choosen as blow.

####run Adam.
```shell
python gan_realdata4.py \
	--optimizer_name adam --learning_rate 0.00001 
```
####run AMSGrad.
```shell
python gan_realdata4.py \
	--optimizer_name amsgrad --learning_rate 0.00001 
```

####run AdaShift with max operation.
```shell
python gan_realdata4.py \
	--optimizer_name adashift --learning_rate 0.0002 
```


### Training Result
All datas of result are stored in directory ```"WGAN/result/ "```.
Every sub directory is named as:
```"run time ID"+"_"+"optimizer name"+"_"+"operation on previous gradients"+"_"+"average window size"+"_"+"learning rate"+"_"+"beta1"+"_"+"beta2"```.

In each subdirectory, ```log.pki``` file stores all raw training results.



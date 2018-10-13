# Densenet on Tiny-imagenet

##Requirements

* Python 3.5
* TF-1.9

##Getting started

###Preparing data
Download tiny-imagenet dataset from [here](https://drive.google.com/open?id=1fmn8Gnbt1IenxXTS66CyRSrlwkTZWBJn) and put it under directory ```./dataset```.
```
./dataset
    tiny-imagenet
        tiny-imagenet_image_matrix.npy
        tiny-imagenet_label_matrix.npy
        tiny-imagenet_val_image_matrix.npy
        tiny-imagenet_val_label_matrix.npy
```

###Training
adam optimizer:
```shell
python run_dense_net.py \
    --train --test \
    --growth_rate=12 \
    --depth=28 \
    --dataset=Tiny \
    --exp_name adam \
    --optimizer=adam \
    --lr=0.001 \
    --beta1=0.9
```
amsgrad optimizer:
```shell
python run_dense_net.py \
    --train --test \
    --growth_rate=12 \
    --depth=28 \
    --dataset=Tiny \
    --exp_name amsgrad \
    --optimizer=amsgrad \
    --lr=0.001 \
    --beta1=0.9
```
adashift optimzier:
```shell
python run_dense_net.py \
    --train --test \
    --growth_rate=12 \
    --depth=28 \
    --dataset=Tiny \
    --batch_size 64 \
    --exp_name adashift \
    --optimizer=adaShift \
    --lr=0.010 \
    --beta1=0.9 \
    --keep_num 30
```

One can use Tensorboard to view the summary during training.
```shell
tensorboard --port 22221 --logdir ./logs
```
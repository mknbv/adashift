# Neural Machine Translation

For this experiment we use small-scale IWSLT15 Vietnames to English translation data 
(the [script](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/download_iwslt15.sh) for downloading). 

We add AMSGrad abd AdaShift optimizers to [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) system
(to do it you nedd to add two more methods in file
[optimizers.py](https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/optimizers.py#L197)). 

## Preprocessing:

Train set:
```
python preprocess.py -train_src iwslt15.en-vi/train.vi -train_tgt iwslt15.en-vi/train.en \
-valid_src iwslt15.en-vi/tst2012.vi -valid_tgt iwslt15.en-vi/tst2012.en -save_data iwslt15.en-vi/iwslt15
```

## Training:
```
python train.py -data ../iwslt15 -src_word_vec_size 512 -tgt_word_vec_size 512 -encoder_type brnn -layers 2 \
-rnn_size 512 -batch_size 128 -train_steps 15000 -optim adashift -dropout 0.2 -adam_beta1 0 -adam_beta2 0.9 \
-learning_rate 0.01 -gpu_ranks 0 -save_checkpoint_steps  1000 -save_model adashift_model

python train.py -data ../iwslt15 -src_word_vec_size 512 -tgt_word_vec_size 512 -encoder_type brnn -layers 2 \
-rnn_size 512 -batch_size 128 -train_steps 15000 -optim adam -dropout 0.2 -adam_beta1 0 -adam_beta2 0.9 \
-learning_rate 0.0001 -gpu_ranks 0 -save_checkpoint_steps  1000 -save_model adam_model

python train.py -data ../iwslt15 -src_word_vec_size 512 -tgt_word_vec_size 512 -encoder_type brnn -layers 2 \
-rnn_size 512 -batch_size 128 -train_steps 15000 -optim amsgrad -dropout 0.2 -adam_beta1 0 -adam_beta2 0.9 \
-learning_rate 0.0001 -gpu_ranks 0 -save_checkpoint_steps  1000 -save_model amsgrad_model

python train.py -data ../iwslt15 -src_word_vec_size 512 -tgt_word_vec_size 512 -encoder_type brnn -layers 2 \
-rnn_size 512 -batch_size 128 -train_steps 15000 -optim adam -dropout 0.2 -adam_beta1 0.9 -adam_beta2 0.999 \
-learning_rate 0.0001 -gpu_ranks 0 -save_checkpoint_steps  1000 -save_model adam_beta_model

```

## Testing:
```
python translate.py  -model adam_step_15000.pt -src ../tst2013.vi -tgt ../tst2013.en \
-report_bleu -beam_size 10  -batch_size 128 -output  pred_adam.txt
```

## Results
![res](https://raw.githubusercontent.com/MichaelKonobeev/adashift/master/img/nmt_adam.png)

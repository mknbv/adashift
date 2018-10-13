# Neural Machine Translation

## Requirements

* Python 3.5
* TF-1.9

## Getting started

### Preparing data
``` shell
nmt/scripts/download_iwslt15.sh ./tmp/nmt_data
```

### Training
adam optimizer:
``` shell
python -m nmt.nmt \
    --src=vi --tgt=en \
    --vocab_prefix=./tmp/nmt_data/vocab  \
    --train_prefix=./tmp/nmt_data/train \
    --dev_prefix=./tmp/nmt_data/tst2012 \
    --test_prefix=./tmp/nmt_data/tst2013 \
    --out_dir=./tmp/adam \
    --attention=scaled_luong \
    --batch_size=128 \
    --dropout=0.2 \
    --encoder_type=bi \
    --metrics=bleu \
    --num_layers=2 \
    --num_encoder_layers=2 \
    --num_decoder_layers=2 \
    --num_train_steps=30000 \
    --decay_scheme=self \
    --num_units=512 \
    --optimizer=adam \
    --learning_rate=0.0001 \
    --beta1 0 --beta2 0.9 \
    --infer_mode=beam_search \
    --beam_width=10 
```
amsgrad optimizer:
``` shell
python -m nmt.nmt \
    --src=vi --tgt=en \
    --vocab_prefix=./tmp/nmt_data/vocab \
    --train_prefix=./tmp/nmt_data/train \
    --dev_prefix=./tmp/nmt_data/tst2012 \
    --test_prefix=./tmp/nmt_data/tst2013 \
    --out_dir=./tmp/amsgrad \
    --attention=scaled_luong \
    --batch_size=128 \
    --dropout=0.2 \
    --encoder_type=bi \
    --metrics=bleu \
    --num_layers=2 \
    --num_encoder_layers=2 \
    --num_decoder_layers=2 \
    --num_train_steps=30000 \
    --decay_scheme=self \
    --num_units=512 \
    --optimizer=amsgrad \
    --learning_rate=0.0001 \
    --beta1 0 --beta2 0.9 \
    --infer_mode=beam_search \
    --beam_width=10 
```
adashift optimizer:
``` shell
python -m nmt.nmt \
    --src=vi --tgt=en \
    --vocab_prefix=./tmp/nmt_data/vocab \
    --train_prefix=./tmp/nmt_data/train \
    --dev_prefix=./tmp/nmt_data/tst2012 \
    --test_prefix=./tmp/nmt_data/tst2013 \
    --out_dir=./tmp/adashift \
    --attention=scaled_luong \
    --batch_size=128 \
    --dropout=0.2 \
    --encoder_type=bi \
    --metrics=bleu \
    --num_layers=2 \
    --num_encoder_layers=2 \
    --num_decoder_layers=2 \
    --num_train_steps=30000 \
    --decay_scheme=self \
    --num_units=512 \
    --optimizer=adaShift \
    --learning_rate=0.0100 \
    --beta1 0 --beta2 0.9 \
    --keep_num 40 \
    --use_mov 1 \
    --mov_num 40 \
    --infer_mode=beam_search \
    --beam_width=10 
```

One can use Tensorboard to view the summary during training.
``` shell
tensorboard --port 22220 --logdir ./tmp
```

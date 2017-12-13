# SampleRNN (ZVK fork)

Code accompanying the paper [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://openreview.net/forum?id=SkxKPDv5xl). Samples are available [here](https://soundcloud.com/samplernn/sets).

## Features of the ZVK fork:

- auto-preprocessing (audio conversion, concatenation, chunking, and saving .npy files)
- generate scripts for trained datasets
- scripts for different sample rates are available (16k, 32k)
- any processed datasets can be loaded into the two-tier network via arguments
- sampling is picked from distribution (not max)
- any number of RNN layers is now possible (until you run out of memory)

## Dependencies
- cuDNN 5105
- Python 2.7.12
- Numpy 1.11.1
- Theano 0.9.0rc3 or 1.0
- Lasagne 0.2.dev1

## Datasets
To preprocess audio for a 32k new experiment, place your audio here:
```
datasets/music/downloads/
```
then run the new experiment python script located in the datasets/music directory:

```
cd datasets/music/
python new_experiment32k.py your_datasets_name downloads/
```

## Training
To train a model on an existing dataset with accelerated GPU processing, you need to run following lines from the root of `sampleRNN_ICLR2017` folder which corresponds to the best found set of hyper-paramters.

Mission control center:
```
$ pwd
/root/zvk/sampleRNN_ICLR2017
```
### SampleRNN (2-tier)
```
$ python models/two_tier/two_tier32k.py -h
usage: two_tier.py [-h] [--exp EXP] --n_frames N_FRAMES --frame_size
                   FRAME_SIZE --weight_norm WEIGHT_NORM --emb_size EMB_SIZE
                   --skip_conn SKIP_CONN --dim DIM --n_rnn {1,2,3,4,5}
                   --rnn_type {LSTM,GRU} --learn_h0 LEARN_H0 --q_levels
                   Q_LEVELS --q_type {linear,a-law,mu-law} --which_set
                   {ONOM,BLIZZ,MUSIC} --batch_size {64,128,256} [--debug]
                   [--resume]

two_tier.py No default value! Indicate every argument.

optional arguments:
  -h, --help            show this help message and exit
  --exp EXP             Experiment name
  --n_frames N_FRAMES   How many "frames" to include in each Truncated BPTT
                        pass
  --frame_size FRAME_SIZE
                        How many samples per frame
  --weight_norm WEIGHT_NORM
                        Adding learnable weight normalization to all the
                        linear layers (except for the embedding layer)
  --emb_size EMB_SIZE   Size of embedding layer (0 to disable)
  --skip_conn SKIP_CONN
                        Add skip connections to RNN
  --dim DIM             Dimension of RNN and MLPs
  --n_rnn {1,2,3,4,5,6,7,8,9,10,11,12,n,...}
					 	Number of layers in the stacked RNN
  --rnn_type {LSTM,GRU}
                        GRU or LSTM
  --learn_h0 LEARN_H0   Whether to learn the initial state of RNN
  --q_levels Q_LEVELS   Number of bins for quantization of audio samples.
                        Should be 256 for mu-law.
  --q_type {linear,a-law,mu-law}
                        Quantization in linear-scale, a-law-companding, or mu-
                        law compandig. With mu-/a-law quantization level shoud
                        be set as 256
  --which_set WHICH_SET any preprocessed set in the datasets/music/ directory
  --batch_size {64,128,256}
                        size of mini-batch
  --debug               Debug mode
  --resume              Resume the same model from the last checkpoint. Order
                        of params are important. [for now]
```
To run:
```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python -u models/two_tier/two_tier32.py --exp BEST_2TIER --n_frames 64 --frame_size 16 --emb_size 256 --skip_conn False --dim 1024 --n_rnn 3 --rnn_type GRU --q_levels 256 --q_type linear --batch_size 128 --weight_norm True --learn_h0 True --which_set user_dataset_name
```
### SampleRNN (3-tier)
```
$ python models/three_tier/three_tier.py -h
usage: three_tier16k.py [-h] [--exp EXP] --seq_len SEQ_LEN --big_frame_size
                     BIG_FRAME_SIZE --frame_size FRAME_SIZE --weight_norm
                     WEIGHT_NORM --emb_size EMB_SIZE --skip_conn SKIP_CONN
                     --dim DIM --n_rnn {1,2,3,4,5} --rnn_type {LSTM,GRU}
                     --learn_h0 LEARN_H0 --q_levels Q_LEVELS --q_type
                     {linear,a-law,mu-law} --which_set {ONOM,BLIZZ,MUSIC}
                     --batch_size {64,128,256} [--debug] [--resume]

three_tier.py No default value! Indicate every argument.

optional arguments:
  -h, --help            show this help message and exit
  --exp EXP             Experiment name
  --seq_len SEQ_LEN     How many samples to include in each Truncated BPTT
                        pass
  --big_frame_size BIG_FRAME_SIZE
                        How many samples per big frame in tier 3
  --frame_size FRAME_SIZE
                        How many samples per frame in tier 2
  --weight_norm WEIGHT_NORM
                        Adding learnable weight normalization to all the
                        linear layers (except for the embedding layer)
  --emb_size EMB_SIZE   Size of embedding layer (> 0)
  --skip_conn SKIP_CONN
                        Add skip connections to RNN
  --dim DIM             Dimension of RNN and MLPs
  --n_rnn {1,2,3,4,5}   Number of layers in the stacked RNN
  --rnn_type {LSTM,GRU}
                        GRU or LSTM
  --learn_h0 LEARN_H0   Whether to learn the initial state of RNN
  --q_levels Q_LEVELS   Number of bins for quantization of audio samples.
                        Should be 256 for mu-law.
  --q_type {linear,a-law,mu-law}
                        Quantization in linear-scale, a-law-companding, or mu-
                        law compandig. With mu-/a-law quantization level shoud
                        be set as 256
  --which_set WHICH_SET
                        any preprocessed set in the datasets/music/ directory
  --batch_size {64,128,256}
                        size of mini-batch
  --debug               Debug mode
  --resume              Resume the same model from the last checkpoint. Order
                        of params are important. [for now]
```
To run:
```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python -u models/two_tier/two_tier32k.py --exp BEST_2TIER --seq_len 512 --big_frame_size 8 --frame_size 2 --emb_size 256 --skip_conn False --dim 1024 --n_rnn 1 --rnn_type GRU --q_levels 256 --q_type linear --batch_size 128 --weight_norm True --learn_h0 True --which_set your_dataset_name
```

To generate 5 sequences (10 seconds each) from a trained model:
```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python -u models/two_tier/two_tier_generate32k.py --exp BEST_2TIER --seq_len 512 --big_frame_size 8 --frame_size 2 --emb_size 256 --skip_conn False --dim 1024 --n_rnn 1 --rnn_type GRU --q_levels 256 --q_type linear --batch_size 128 --weight_norm True --learn_h0 True --which_set your_dataset_name --n_secs 10 --n_seqs 5
```

## Reference
If you are using this code, please cite the paper.

SampleRNN: An Unconditional End-to-End Neural Audio Generation Model. Soroush Mehri, Kundan Kumar, Ishaan Gulrajani, Rithesh Kumar, Shubham Jain, Jose Sotelo, Aaron Courville, Yoshua Bengio, 5th International Conference on Learning Representations (ICLR 2017), submitted and under review.

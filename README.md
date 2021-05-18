# SSL-ECG-Paper-Reimplementaton
Reimplementation of the paper [Sarkar and Etemad 2020 "Self-Supervised ECG Representation Learning for Emotion Recognition"](https://ieeexplore.ieee.org/document/9161416)

There exist a Tensorflow 1.14 version from the original author in [this repository](https://code.engineering.queensu.ca/17ps21/SSL-ECG), however the converters to the datasets are missing

## Setup

I use anaconda, and the guide assumes you use that too. In case you don't, I guess it is not too hard to modify the `install.sh` script to your requirenments.

Anyway, I provided two possible ways to get your conda env working:

`conda env create -n conda-env -f sslecg.yml`

or you execute the `install.sh` script, although you need to manually intervine all the times.

**After you did install the environment:**

After you did install the environment, there are some more steps to do:

create a folder `cache` on the top level, and then one for each dataset you want to use inside, so `cache/amigos`, 
`cache/wesad` and `cache/dreamer`.

also edit the file `src/constants.py` and change the variables towards the correct 
## Mode of usage

I uploaded the pretrained models, so you can use them like in `src/run_example.py`

If you want to use the pretraining, you also must download each of the datasets:
- [AMIGOS](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/download.html)
- [DREAMER](https://zenodo.org/record/546113#.YKLY0WYzaHs)
- [WESAD](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)
- [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/)

All of them must be in a common folder, but you can configure this basepath in `src/constants.py`

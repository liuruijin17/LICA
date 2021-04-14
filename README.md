**LICA**: A Lightweight Lane Shape Detector with Curvature-Aware Learning
=======

* ⚡⚡ Super lightweight: The number of model parameters is only 267,867.
* ⚡⚡ Super low complexity: The number of MACs (1 MAC = 2 FLOP) is only 486.617M.
* 😎  Learning structures with large curvatures without expensive, dense, and precise human annotations.


## Model Zoo
The pretrained models are stored in LICAZoos/

## Set Envirionment

* Linux ubuntu 16.04
* GeForce RTX 3090
* Python 3.8.5
* CUDA 11.1

Create virtualenv environment

```
python3 -m venv lica
```

Activate it

```
source lica/bin/activate
```

Then install dependencies

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Data Preparation
Download and extract [TuSimple](https://github.com/TuSimple/tusimple-benchmark),
and [LLAMAS](https://unsupervised-llamas.com/llamas/download).

We expect the directory structure to be the following:
```
lica/
LICA/
    LICA/
    LICAZoos/
TuSimple/
    LaneDetection/
        clips/
        label_data_0313.json
        label_data_0531.json
        label_data_0601.json
        test_label.json
LLAMAS/
    color_images/
    labels/
```

## Evaluation


### TuSimple:

#### LICA(TR2)
```
python test.py LICA_TR2_TUSIMPLE --testiter 500000
```

#### LICA(TR6)
```
python test.py LICA_TR6_TUSIMPLE --testiter 500000
```

### LLAMAS:

#### LICA(TR2)
```
python test.py LICA_TR2_LLAMAS --testiter 500000 --split validation
```

## Training

Corresponding codes will be released after acceptance.

## Acknowledgements

[LSTR](https://github.com/liuruijin17/LSTR)

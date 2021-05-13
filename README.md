**LICA**: A Lightweight Lane Shape Detector with Curvature-Aware Learning
=======

* ⚡⚡ Super lightweight: The number of model parameters is 267,867 (< LSTR's 765,787).
* ⚡⚡ Super low complexity: The number of MACs (1 MAC = 2 FLOP) is only 486.617M (< LSTR's 574.280M).
* 😎  Learning structures with large curvatures without expensive, dense, and precise human annotations (predicting
  curved lanes better than LSTR).

## Demo

The LLAMAS dataset paper defined and required lane categories when evaluating algorithms.
l0 means the closest left lane to the host-vehicle, r0 means the closest right lane (likewise for l1 and r1).

In these demo videos, no tracking module is used. 

### Comparison between LICA and LI. 
The LI is trained without the proposed curvature-aware learning.

[![LICA LI LLAMAS video](http://img.youtube.com/vi/SjBGOYSgisU/0.jpg)](http://www.youtube.com/watch?v=SjBGOYSgisU "LICA LI LLAMAS video")

### Comparison between LICA and LSTR.
[![LICA LSTR LLAMAS video](http://img.youtube.com/vi/6cHtrE3ZsUQ/0.jpg)](http://www.youtube.com/watch?v=6cHtrE3ZsUQ "LICA LSTR LLAMAS video")


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

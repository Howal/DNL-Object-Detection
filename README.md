# DNLNet for Object Detection

By Minghao Yin, Zhuliang Yao, Yue Cao, Xiu Li, Zheng Zhang, Stephen Lin, Han Hu.

This repo is a official implementation of ["Disentangled Non-Local Neural Networks"](https://arxiv.org/abs/2006.06668) on COCO object detection based on open-mmlab's [mmdetection](https://github.com/open-mmlab/mmdetection).
Many thanks to mmdetection for their simple and clean framework.


## Introduction

**DNLNet** is initially described in [arxiv](https://arxiv.org/abs/2006.06668). It is still in progress.

## Citing DNLNet

```
@misc{yin2020disentangled,
    title={Disentangled Non-Local Neural Networks},
    author={Minghao Yin and Zhuliang Yao and Yue Cao and Xiu Li and Zheng Zhang and Stephen Lin and Han Hu},
    year={2020},
    booktitle={ECCV}
}
```

## Main Results

### Results on R50-FPN with backbone (syncBN)

|  Back-bone |   Model   | Backbone Norm |       Heads      |     Context    | Lr schd | box AP | mask AP | Download |
|:---------:|:---------:|:-------------:|:----------------:|:--------------:|:-------:|:-------:|:--------:|:--------:|
|  R50-FPN |       Mask       |     SyncBN    |    4Conv1FC   |        -       |    1x   | 38.8  |   35.1  | [model](https://microsoft-my.sharepoint.com/:u:/p/t-zhuyao/ESBsU_-lvpxLvwgdEA3sKnABUPP6MPJbVShdLyvfFvtdgg?e=x9FTqm) &#124; [log](https://microsoft-my.sharepoint.com/:u:/p/t-zhuyao/EfgX4VrNMEhLgRqzcIKEOf8BkbBk5fNwA5d9fLL5xC2KXQ?e=pHrVrF) |
|  R50-FPN |       Mask       |     SyncBN    |    4Conv1FC   | NL(c4) |    1x   | 39.6  |   35.8  | [model](https://microsoft-my.sharepoint.com/:u:/p/t-zhuyao/EaRVsOJnGNpKu7NdARGBt9gBZQ5OU2X8RzoWBiq1A25BZA?e=4aQeL0) &#124; [log](https://microsoft-my.sharepoint.com/:u:/p/t-zhuyao/EQkuCp1hNDtNsgKNSy3bs2cBgD3Ygi3nKImKySeiNrXFng?e=ockp3g) |
|  R50-FPN |       Mask       |     SyncBN    |    4Conv1FC   | GC(c4, r4) |    1x   | 40.1  |   36.2  | [model](https://microsoft-my.sharepoint.com/:u:/p/t-zhuyao/EVmIm_qTamRHtG2EYek6kOIB_aCgIqmEuxeXCSiZTncxhw?e=ROfDMV) &#124; [log](https://microsoft-my.sharepoint.com/:u:/p/t-zhuyao/ERqhLMv0hNlPgycna57dolIBATwuGrHgxvogZbOXwf2rkg?e=rkXZbT) |
|  R50-FPN |       Mask       |     SyncBN    |    4Conv1FC   | SNL(c4) |    1x   | 40.1  |   36.2  | [model](https://microsoft-my.sharepoint.com/:u:/p/t-zhuyao/EaLgMNGziWRLhGmFTdeSIHwB8XtxTDhqTVGvw1drmo_bhw?e=c9a07t) &#124; [log](https://microsoft-my.sharepoint.com/:u:/p/t-zhuyao/Ee1vsUBdCuFEnHaacapJAYMBtPQui2h4wE4POV5U5fDMww?e=2Wcu6l) |
|  R50-FPN |       Mask       |     SyncBN    |    4Conv1FC   | DNL(c4) |    1x   | 40.3  |   36.4  | [model](https://microsoft-my.sharepoint.com/:u:/p/t-zhuyao/EUJfTYRitsdDmkSsdVc7eGcBKQQMXqd8qvl144juAsxqrw?e=iPp4jO) &#124; [log](https://microsoft-my.sharepoint.com/:u:/p/t-zhuyao/ESMinReWPuNAqJs4J-6xHAsBHp8pwT8bORt_3TDgENhOhg?e=PIGbSM) |
|  R50-FPN |       Mask       |     SyncBN    |    4Conv1FC   | DNL(c4+c5_all) |    1x   | 41.2  |   37.2  | [model](https://microsoft-my.sharepoint.com/:u:/p/t-zhuyao/EQilrnU8FSZPmKAP99YqgIYBwRUOq2ChWMNrgqMlcaxevw?e=GcgBV1) &#124; [log](https://microsoft-my.sharepoint.com/:u:/p/t-zhuyao/EX_sejC2I-1NsLwTPW2FYVcBHgKoTs62qjTOzLEcYbOV3g?e=tnVyYu) |

### Results on stronger backbones
- On going


**Notes**
- `NL` denotes Non-local block block is inserted after 1x1 conv of backbone.
- `GC` denotes Global Context (GC) block is inserted after 1x1 conv of backbone.
- `SNL` denotes Simplified Non-local block block is inserted after 1x1 conv of backbone.
- `DNL` denotes Disentangled Non-local block block is inserted after 1x1 conv of backbone.
- `r4` denotes ratio 4 in GC block.
- `c4` and `c5_all` denote insert context block at stage c4's last residual block and c5's all blocks, respectively.
- Most models are trained on 16 GPUs with 4 images on each GPU.

## Requirements

- Linux(tested on Ubuntu 16.04)
- Python 3.6+
- Cython
- PyTorch 1.1.0
- CUDA 9.0
- CUDNN 7.0
- NCCL 2.3.5
- [apex](https://github.com/NVIDIA/apex)
- [inplace_abn](https://github.com/mapillary/inplace_abn)

## Install

a. Install PyTorch 1.1 and torchvision following the [official instructions](https://pytorch.org/).

b. Install latest apex with CUDA and C++ extensions following this [instructions](https://github.com/NVIDIA/apex#quick-start). 
The [inplace_abn](https://github.com/mapillary/inplace_abn) implemented by apex is required.

c. Clone the DNLNet repository. 

```bash
 git clone https://github.com/Howal/DNL-Object-Detection.git
```

d. Install DNLNet version mmdetection (other dependencies will be installed automatically).

```bash
python(3) setup.py build develop  # add --user if you want to install it locally
# or "pip install -e -v ."
```

Please refer to mmdetection install [instruction](https://github.com/open-mmlab/mmdetection/blob/master/INSTALL.md) for more details.


## Usage

### Train

As in original mmdetection, distributed training is recommended for either single machine or multiple machines.

```bash
./tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> [optional arguments]
```

Supported arguments are:

- --validate: perform evaluation every k (default=1) epochs during the training.
- --work_dir <WORK_DIR>: if specified, the path in config file will be replaced.

### Evaluation

To evaluate trained models, output file is required.

```bash
python tools/test.py <CONFIG_FILE> <MODEL_PATH> [optional arguments]
```

Supported arguments are:

- --gpus: number of GPU used for evaluation
- --out: output file name, usually ends wiht `.pkl`
- --eval: type of evaluation need, for mask-rcnn, `bbox segm` would evaluate both bounding box and mask AP. 

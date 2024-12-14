# Towards Understanding Evolving Patterns in Sequential Data

[__[Paper]__](https://openreview.net/pdf?id=I2gVmVRgNk) 
&nbsp; 
This is the authors' official PyTorch implementation for SDE-EDG method in the **NeurIPS 2024 (Spotlight)** paper [Towards Understanding Evolving Patterns in Sequential Data](https://openreview.net/pdf?id=I2gVmVRgNk).


## Prerequisites
- PyTorch 
- Python3
- Numpy
- pandas 
- matplotlib
- seaborn

## Dataset
- Circle/Sine/RMNIST/Portraits/Caltran/PowerSupply download through [here](https://github.com/WonderSeven/LSSAE?tab=readme-ov-file)
- [Ocular Disease](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k/data)

## Run EvoRate with Correspondences
```
python with-correspondence/main.py 
```

## Run EvoRate without Correspondences
```
python without-correspondence/main.py 
```


## Citation
If you use this code for your research, please consider citing:
```
@inproceedings{
zeng2024towards,
title={Towards Understanding Evolving Patterns in Sequential Data},
author={QIUHAO Zeng and Long-Kai Huang and Qi CHEN and Charles Ling and Boyu Wang},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=I2gVmVRgNk}
}
```
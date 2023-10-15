# Self-DACE LLIE Method
Official pytorch version for Self-Reference Deep Adaptive Curve Estimation for Low-Light Image Enhancement

- Paper is avalible at [arXiv version - Self-Reference Deep Adaptive Curve Estimation for Low-Light Image Enhancement](https://arxiv.org/pdf/2308.08197.pdf)

# Demo
## Demo on Low-light Images
![demo_1](demo4git/demo1.png) | ![demo_2](demo4git/demo2.png) 
---|---
![demo_3](demo4git/demo3.png) | ![demo_4](demo4git/demo4.png) 

Visual comparison with original low-light image
on LOL and SCIE dataset. The enhanced images of our
method are on the top-right corners, and the input low-light
images are on the bottom-left corners.

## Demo on Improvement of Low-light Image Segmentation
![demo_1](visualization/vis1.jpg)
![demo_1](visualization/vis2.jpg)
Demostration of improvement for segmentation task on Dark Face Dataset using PiClick.
The green stars are the objects of interactive segmentation what we want to segment.
GT is annotated manually by us.

# Comparison
## Table 1. Quantitative comparisons in terms of four full-reference image quality metrics including PSNR(dB), SSIM, LPIPS and CIEDE2000 on the LOL test, LSRWand SCIE Part2 datasets.
![metrics](demo4git/com1.png) 

Ours* is the result only from Stage-I.

### Note:
- Our metric values are in line with [Low-Light Image and Video Enhancement Using Deep Learning: A Survey](https://github.com/Li-Chongyi/Lighting-the-Darkness-in-the-Deep-Learning-Era-Open).

## Table 2. Comparisons of computational complexity in termsof number of trainable parameters and FLOPs.

<div align=center>
<img src="demo4git/com2.png">
</div>

Ours∗ is the model of Stage-I, and Ours is the model including Stage-I and Stage-II. Those are applied to a 3x1200×900 image.

## Visual Comparison on LIME
<div align=center>
<img src="demo4git/visual.png">
</div>

The blue box zooms in the complex light and dark junction of the input image.
Image of Ours∗ is the output only from Stage-I.

# How to use it
## Prerequisite
```
cd ./codes_SelfDACE
pip install -r ./requirements.txt
```

## Test Stage-I (only enhancing luminance)
```
cd ./stage1
python test_1stage.py
```
Test dates should be placed in `codes_SelfDACE/stage1/data/test_data/low_eval`,
And then results would be found in `codes_SelfDACE/stage1/data/result/low_eval`.

## Test both Stage-I and Stage-II (enhancing luminance and denoising)
```
cd ./stage2
python test_1stage.py
```
Test dates should be placed in `codes_SelfDACE/stage2/data/test_data/low_eval`,
And then results would be found in `codes_SelfDACE/stage2/data/result/low_eval`.

# How to train it
## Prerequisite
```
cd ./codes_SelfDACE
pip install -r ./requirements.txt
```

## Train Stage-I (only enhancing luminance)
1.
      You should download the training dataset from [SCIE_part1](https://github.com/csjcai/SICE) and resize all images to 512x512.
      Or you could download it directly from [SCIE_part1_ZeroDCE_version](https://github.com/Developer-Zer0/ZeroDCE), of which iamges have been cropped to 512x512 already. If you want to use it in your work, please cite [SCIE_part1](https://github.com/csjcai/SICE).

2.
      ```
      cd ./stage1
      python train_1stage.py
      ```

## Train Stage-II (only denoising)

1.
      Copy the `pre-trained model` and `training dataset` from stage1, and put `pre-trained model` of Stage-I in `./stage2/snapshots_light`

2.
      ```
      cd ./stage2
      python train_2stage.py
      ```

# Citation
If you find our work useful for your research, please cite our paper
```
@article{wen2023self,
  title={Self-Reference Deep Adaptive Curve Estimation for Low-Light Image Enhancement},
  author={Wen, Jianyu and Wu, Chenhao and Zhang, Tong and Yu, Yixuan and Swierczynski, Piotr},
  journal={arXiv preprint arXiv:2308.08197},
  year={2023}
}
```
# Hope AI would light our unknown and unforeseeable routes to the future as well as human's as ligting the low-light image
- Thanks for all related work and workers.

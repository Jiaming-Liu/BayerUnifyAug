# BayerUnifyAug
Python toolbox for Bayer Raw image unification and augmentation.
This repo implements the data pre-processing part of our NTIRE 2019 Real Image Denoising Challenge submission.
See: http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Liu_Learning_Raw_Image_Denoising_With_Bayer_Pattern_Unification_and_Bayer_CVPRW_2019_paper.html


## Dependency
Python3, Numpy

Note: For Python 2, change the function signatures to Python 2 style.

## Demo
```python
import bayer_unify_aug
import cv2

# load RGGB image
rggb = cv2.imread("./demo/lena_rggb.png", -1)

# output demosaic result
bgr = cv2.cvtColor(rggb, cv2.COLOR_BAYER_BG2BGR)
cv2.imwrite("./demo/lena_direct_out.jpg", bgr)

# convert RGGB to BGGR and then demosaic
bggr = bayer_unify_aug.bayer_unify(
    rggb, input_pattern="RGGB", target_pattern="BGGR", mode="pad")
bgr = cv2.cvtColor(bggr, cv2.COLOR_BAYER_RG2BGR)
cv2.imwrite("./demo/lena_bggr_out.jpg", bgr)

# flip RGGB image and then demosaic
flipped = bayer_unify_aug.bayer_aug(
    rggb, flip_h=False, flip_w=True, transpose=False, input_pattern="RGGB")
bgr = cv2.cvtColor(flipped, cv2.COLOR_BAYER_BG2BGR)
cv2.imwrite("./demo/lena_flipped_out.jpg", bgr)

```

## Citing
If BayerUnify and BayerAug help your research, please cite it using the following BibTex entry:
```
@InProceedings{Liu_2019_CVPR_Workshops,
author = {Liu, Jiaming and Wu, Chi-Hao and Wang, Yuzhi and Xu, Qin and Zhou, Yuqian and Huang, Haibin and Wang, Chuan and Cai, Shaofan and Ding, Yifan and Fan, Haoqiang and Wang, Jue},
title = {Learning Raw Image Denoising With Bayer Pattern Unification and Bayer Preserving Augmentation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}
```

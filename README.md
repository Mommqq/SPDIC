# SPDIC

**Perceptual Distributed Image Compression via Controlled Stochastic Reconstruction**  
Guojun Xu, Jianwen Xiang, Yaning Xie, and Junwei Zhou  
School of Artificial Intelligence

[Project page](https://mommqq.github.io/SPDIC-Demo/)

SPDIC studies distributed image compression when a correlated image is available only at the decoder. Controlled stochastic reconstruction is combined with decoder-only textual and visual side information to recover local variation while retaining target correspondence.

The local Gaussian analysis relates reconstruction scale to target correspondence. The quantity $P_G$ measures the discrepancy between local Gaussian marginals and is distinct from LPIPS.

Experiments on KITTI Stereo and Cityscapes compare bitrate, pixel fidelity, and feature-space similarity. The reconstruction comparisons use native-resolution image pairs and report BPP, PSNR, and LPIPS-VGG.

![Local Gaussian analysis](figure/figure1.png)

![SPDIC architecture](figure/figure2.png)

![Quantitative comparison](figure/figure3.png)

![Reconstruction comparison](figure/figure4.png)

The implementation is available in `modules/` and `train.py`.

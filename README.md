## PIG

This repository contains the official source code for the paper:

"Paired Image Generation with Diffusion-Guided Diffusion Models"

📄 Paper Accepted by: MICCAI 2025

🔗 Paper Link: https://papers.miccai.org/miccai-2025/paper/4386_paper.pdf



## Citation

If you find our work useful, please consider citing:

```
@InProceedings{pig,
author="Zhang, Haoxuan
and Cui, Wenju
and Cao, Yuzhu
and Tan, Tao
and Liu, Jie
and Peng, Yunsong
and Zheng, Jian",
editor="Gee, James C.
and Alexander, Daniel C.
and Hong, Jaesung
and Iglesias, Juan Eugenio
and Sudre, Carole H.
and Venkataraman, Archana
and Golland, Polina
and Kim, Jong Hyo
and Park, Jinah",
title="Paired Image Generation with Diffusion-Guided Diffusion Models",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="371--381",
abstract="The segmentation of mass lesions in digital breast tomosynthesis (DBT) images is very significant for the early screening of breast cancer. However, the high-density breast tissue often leads to high concealment of the mass lesions, which makes manual annotation difficult and time-consuming. As a result, there is a lack of annotated data for model training. Diffusion models are commonly used for data augmentation, but the existing methods face two challenges. First, due to the high concealment of lesions, it is difficult for the model to learn the features of the lesion area. This leads to the low generation quality of the lesion areas, thus limiting the quality of the generated images. Second, existing methods can only generate images and cannot generate corresponding annotations, which restricts the usability of the generated images in supervised training. In this work, we propose a paired image generation method. The method does not require external conditions and can achieve the generation of paired images by training an extra diffusion guider for the conditional diffusion model. During the experimental phase, we generated paired DBT slices and mass lesion masks. Then, we incorporated them into the supervised training process of the mass lesion segmentation task. The experimental results show that our method can improve the generation quality without external conditions. Moreover, it contributes to alleviating the shortage of annotated data, thus enhancing the performance of downstream tasks.",
isbn="978-3-032-04965-0"
}
```


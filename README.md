<div align=center>
<h1> Enhancing Out-of-Distribution Object Detection with CLIP </h1>
</div>
<div align=center>

<div align=center>
<a src="https://img.shields.io/badge/%F0%9F%93%96-Report-red.svg?style=flat-square" href="https://docs.google.com/document/d/1k0BcnrnMjDZfE_vwAiLpe5KyqjaywVXNd_452LeZfjM/edit?usp=sharing">
<img src="https://img.shields.io/badge/%F0%9F%93%96-Report-red.svg?style=flat-square">
</a>

<a src="https://img.shields.io/badge/%F0%9F%8E%A4-Slides-blue.svg?style=flat-square" href="https://brown365-my.sharepoint.com/:p:/g/personal/xma75_ad_brown_edu/EWl8KQBX871LmJq8Lz6NoHMBC3KXe2If3L-jqN38WdxD0w?e=iYzHeW&nav=eyJzSWQiOjI1N30">
<img src="https://img.shields.io/badge/%F0%9F%8E%A4-Slides-blue.svg?style=flat-square">
</a>

</div>

<p align="center">
  <img src="imgs/main_diagram.jpeg" width=800><br/>
</p>

<!-- <p align="center">
  <img src="imgs/output3.png" width=600><br/>
</p> -->

## Introduction

In artificial intelligence, achieving robust performance under unforeseen circumstances is a pivotal challenge. The Out-Of-Distribution (OOD) problem exemplifies this challenge, as it involves text descriptions that do not correlate with the visual elements in the image. This problem arises when a model encounters data that is significantly different from the training set, challenging its ability to make accurate predictions. In the domain of object detection, by giving text descriptions and the image, the model may draw bounding boxes of the text in the image even when the described item doesn't exist in the image. Therefore, we aim to solve this problem by focusing on the out-of-distribution problem in object detection. This initiative is also inspired by advancements from two recent studies: 'CLIPN for Zero-Shot Detection: Teaching CLIP to Say No' and 'RegionCLIP: Region-based Language-Image Pretraining.' The first introduces a decision-making process that allows models to detect OOD and ID samples, and the second enhances the precision of detecting and aligning specific image regions with corresponding text descriptions. In our context, the Out-Of-Domain (OOD) problem in object detection is characterized by the presence of text descriptions that do not correspond to any visual content within the image, and therefore, no bounding box should be delineated for such descriptions. We propose a framework that integrates a decision module within a CLIP-guided region-text alignment mechanism. Our approach specifically addresses detection challenges, determining whether certain regions identified in an image should be marked, despite being described in concept pools. This is critical for scenarios where textual descriptions exist without corresponding visual evidence in the images. By combining detection capabilities with selective decision-making, our model aims to recognize OOD items and accurately detect object using bounding boxes, while enhancing their efficacy and reliability.


## Requirements

## Folder Explanation 

**CLIPN**: 
* A reproduce of [CLIPN for Zero-Shot OOD Detection: Teaching CLIP to Say No](https://arxiv.org/abs/2308.12213) based on [xmed-lab/CLIPN](https://github.com/xmed-lab/CLIPN))


## Ablation Experiments
Results: [ablation_results](https://drive.google.com/drive/folders/1dfPbZx8CwcK3-M88zTtgnMVG9VbzIrAW)
* Transform
* Loss
* Training strategy

## Future Work

**Training & Experiments**
* boost the AUROC with hyperparameter tuning
* Training with Pseudo Labels from RPN

**Algorithms
* 

**Others**
* visualize the outcomes of the object OOD detection model

**Acknowledge**

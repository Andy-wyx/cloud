<div align=center>
<h1> Enhancing Out-of-Distribution Object Detection with CLIP </h1>
</div>

<div align=center>
<a src="https://img.shields.io/badge/%F0%9F%93%96-Report-red.svg?style=flat-square" href="https://docs.google.com/document/d/1k0BcnrnMjDZfE_vwAiLpe5KyqjaywVXNd_452LeZfjM/edit?usp=sharing">
<img src="https://img.shields.io/badge/%F0%9F%93%96-Report-red.svg?style=flat-square">
</a>

<a src="https://img.shields.io/badge/%F0%9F%8E%A4-Slides-blue.svg?style=flat-square" href="https://brown365-my.sharepoint.com/:p:/g/personal/xma75_ad_brown_edu/EWl8KQBX871LmJq8Lz6NoHMBC3KXe2If3L-jqN38WdxD0w?e=iYzHeW&nav=eyJzSWQiOjI1N30">
<img src="https://img.shields.io/badge/%F0%9F%8E%A4-Slides-blue.svg?style=flat-square">
</a>

</div>

<!-- <p align="center">
  <img src="imgs/output3.png" width=600><br/>
</p> -->


## :rocket: Introduction
In the field of artificial intelligence, one of the primary challenges is ensuring that models perform robustly under unexpected conditions. Traditional object detection models, such as Faster R-CNN, struggle with detecting __Out-Of-Distribution (OOD)__ data, often leading to incorrect detections or misclassifications. Inspired by [CLIPN](https://arxiv.org/abs/2308.12213), we developed the ***CLOUD***-Contrastive Learning Based Out-of-Distribution Unified Detector. This model applies OOD detection capability in multi-object detection scenarios.  
Our contribution includes
- the creation of a new dataset
- the development of a joint training pipeline
- and the implementation of region-text matching techniques alongside new loss strategies to enhance overall model performance


## :cloud: CLOUD: Contrastive Learning based Out-of-distribution Unified Detector.
<p align="center">
  <img src="imgs/main_diagram.jpeg" width=800><br/>
</p>

## Installation


## Prepare Dataset


## Ablation Experiments
Results: [ablation_results](https://drive.google.com/drive/folders/1dfPbZx8CwcK3-M88zTtgnMVG9VbzIrAW)
* Transform
* Loss
* Training strategy

## Training Results

## Visualization

## Future Work


**Acknowledge**
* [xmed-lab/CLIPN](https://github.com/xmed-lab/CLIPN)
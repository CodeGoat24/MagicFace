<div align="center">

<h1>MagicFace: Training-free Universal-Style Human Image Customized Synthesis</h1>

[Yibin Wang](https://codegoat24.github.io)\*, [Weizhong Zhang](https://weizhonz.github.io/)\*, [Cheng Jin](https://cjinfdu.github.io/)&#8224; 

(*equal contribution, &#8224;corresponding author)

[Fudan University]

<a href="https://arxiv.org/pdf/2408.07433">
<img src='https://img.shields.io/badge/arxiv-MagicFace-blue' alt='Paper PDF'></a>
<a href="https://codegoat24.github.io/MagicFace/">
<img src='https://img.shields.io/badge/Project-Website-orange' alt='Project Page'></a>

</div>

![teaser](docs/static/images/teaser.png)

## Release
- [2024/08/20] 🔥 We update the figures and include inference time comparisons in the [paper](https://arxiv.org/pdf/2408.07433).
- [2024/08/15] 🔥 We release the [paper](https://arxiv.org/pdf/2408.07433).
- [2024/08/14] 🔥 We launch the [project page](https://codegoat24.github.io/MagicFace/).

## 📖 Abstract

<p>
Current human image customization methods leverage Stable Diffusion (SD) for its rich semantic prior. 
However, since SD is not specifically designed for human-oriented generation, these methods often require extensive fine-tuning on large-scale datasets, which renders them susceptible to overfitting and hinders their ability to personalize individuals with previously unseen styles.
Moreover, these methods extensively focus on single-concept human image synthesis and lack the flexibility to customize individuals using multiple given concepts, thereby impeding their broader practical application.
This paper proposes MagicFace, a novel training-free method for multi-concept universal-style human image personalized synthesis. 
Our core idea is to simulate how humans create images given specific concepts, i.e., first establish a semantic layout considering factors such as concepts' shape and posture, then optimize details by comparing with concepts at the pixel level. To implement this process, we introduce a coarse-to-fine generation pipeline, involving two sequential stages: semantic layout construction and concept feature injection. This is achieved by our Reference-aware Self-Attention (RSA) and Region-grouped Blend Attention (RBA) mechanisms.  
In the first stage, RSA enables the latent image to query features from all reference concepts simultaneously, extracting the overall semantic understanding to facilitate the initial semantic layout establishment. 
In the second stage, we employ an attention-based semantic segmentation method to pinpoint the latent generated regions of all concepts at each step. Following this, RBA divides the pixels of the latent image into semantic groups, with each group querying fine-grained features from the corresponding reference concept.
Notably, our method empowers users to freely control the influence of each concept on customization through a weighted mask strategy.
Extensive experiments demonstrate the superiority of MagicFace in both single- and multi-concept human image customization. 
</p>

![architecture](docs/static/images/architecture.png)

## 🗓️ TODO
- [ ] Release inference code
- [ ] Release demo
- [ ] Release evaluation datasets
- [ ] Release evaluation code
- [ ] Release visualization code

## 🖼️ Visual results of MagicFace
![visual_result](docs/static/images/visual_result_photorealism.png)

![visual_result](docs/static/images/visual_result_diverse_style.png)

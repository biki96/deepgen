

<p align="center">
  <img src="figure/logo.jpg" alt="DeepGen" width="450"/ hei>
</p>

<h1 align="center">DeepGen 1.0: A Lightweight Unified Multimodal
Model for Advancing Image Generation and Editing
</h1>

<p align="center">
Shanghai Innovation Institut, DeepGen Team
</p>

<p align="center">
<a href="http://arxiv.org/abs/2602.12205">
<img src='https://img.shields.io/badge/arXiv-DeepGen-blue' alt='Paper PDF'></a>

<a href="https://github.com/deepgenteam/deepgenteam.github.io/">
<img src='https://img.shields.io/badge/Website-project page-orange' alt='Project Page'></a>

<a href="https://github.com/deepgenteam/deepgen_rl">
<img src='https://img.shields.io/badge/GitHub-DeepGen--RL-black' alt='DeepGen RL'></a>

<a href="https://huggingface.co/deepgenteam/DeepGen-1.0">
<img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepGen-yellow' alt='Model CkPT'></a>

<a href="https://huggingface.co/deepgenteam/DeepGen-1.0">
<img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data Coming Soon-yellow' alt='Data'></a>
</p>


## 🔥 News
- **Feb 13, 2026:** We released **DeepGen 1.0**, Pre-traning, Supervised Fine-Tuning and Reinforcement Learning checkpoints can be found in [Huggingface](https://huggingface.co/deepgenteam/DeepGen-1.0), support both T2I generation and image editing.
- **Feb 13, 2026:** We released the training code support Pre-traning, Supervised Fine-Tuning, Reinforcement Learning(deepgen_rl) and evaluation code support wide range of benchmarks.
- **Feb 13, 2026:** We released the **DeepGen 1.0** technical report on [Arxiv](http://arxiv.org/abs/2602.12205)




## ✨ Introduction
**Broader Scenario and Dimension Coverage**
We propose DeepGen 1.0, a lightweight unified multimodal model with only 5B parameters (3B VLM + 2B DiT). It integrates five core capabilities—general image generation, general image editing, reasoning image generation, reasoning image editing, and text rendering—within a single model. Across multiple authoritative benchmarks, DeepGen 1.0 is competitive with competitive with or surpassing the state-of-the-art unified multimodal models that are 3× to 16× larger, achieving comprehensive performance, demonstrating that massive scaling is not the sole path to high-performance multimodal generation.

<p align="center"><img src="figure/bubble_chart.png" width="95%"></p>




## 🧠 Method
Our core observation is that a lightweight model, when empowered by synergistic architecture design and data-centric training strategies, can achieve comprehensive capabilities competitive with or even surpassing much larger counterparts.
To overcome the limitations of lightweight models in semantic understanding and fine-grained control, we introduce **Stacked Channel Bridging (SCB)**, a deep alignment framework that extracts hierarchical features from multiple VLM layers and fuses them with learnable ``think tokens'' to provide the generative backbone with structured, reasoning-rich guidance. 
We further design a data-centric training strategy spanning three progressive stages: (1) **Alignment Pre-training** on large-scale image-text pairs and editing triplets to synchronize VLM and DiT representations, (2) **Joint Supervised Fine-tuning** on a high-quality mixture of generation, editing, and reasoning tasks to foster omni-capabilities, and (3) **Reinforcement Learning with MR-GRPO**, which leverages a mixture of reward functions and supervision signals, resulting in substantial gains in generation quality and alignment with human preferences, while maintaining stable training progress and avoiding visual artifacts.

<p align="center"><img src="figure/arch.png" width="80%"></p>


## 🔥 Set Up Environment
```
conda create -n deepgen python=3.10 -y
conda activate deepgen
```

## 🔧 Checkpoint Preparation
1. DeepGen Checkpoint Preparation
```
huggingface-cli download --resume-download deepgen --local-dir ./ckpt
```

## 📑 Prompt Introduction

- **image_path**: ...



## 🚀 Inference
```
test
```

## ✨ Evaluation
```
test
```

## 💻 Training

#### 1. Checkpoint Download
```
test
```

#### 2. Prepare Training Data
```
test
```

#### 3. Train

## 📧 Contact


## ⭐ Citation
```bibtex
```


# PENet：Progressive Expansion for Semi-supervised Bi-modal Salient Object Detection
The paper has been accepted for publication in the journal Pattern Recognition.

Comparison of architectures and qualitative and quantitative analyses of existing fully supervised, weakly supervised, and proposed semi-supervised SOD models.
---
<p align="center">
  <img src="https://github.com/user-attachments/assets/32b25700-02a7-46d5-a352-ad2a81c53ee8" width="45%" style="display:inline; margin-right:10px;" />
  <img src="https://github.com/user-attachments/assets/25ad0515-5f54-42fa-8220-0f18e7637c99" width="45%" style="display:inline;" />
</p>

Network Architecture
---
![fig 3](https://github.com/user-attachments/assets/ebebaabe-e236-41d7-b56a-8a8293dea5ae)

Quantitative and qualitative comparison with SOTA methods
---
We provide saliency prediction results and the saved models on both RGB-T and RGB-D tasks. [Baidu Pan link:](https://pan.baidu.com/s/1_T8b9eCjVE0oaCvD_jRhJw)    code: 0825

RGB-T SOD:
<p align="center">
  <img src="https://github.com/user-attachments/assets/6a3f5ac7-3dd4-42b2-804b-49cc45ea207c" width="45%" style="display:inline; margin-right:10px;" />
  <img src="https://github.com/user-attachments/assets/29a9c096-91de-4a0b-a8d3-c5f775fbda6a" width="45%" style="display:inline;" />
</p>

![visual1](https://github.com/user-attachments/assets/ec5b4cbc-be99-458b-a164-e712fe5f841f)

RGB-D SOD:
![RGBD](https://github.com/user-attachments/assets/7bfb9374-4a78-4cc6-98c3-1a310e2e729c)
![visual](https://github.com/user-attachments/assets/ee829030-2b8c-4767-a2fc-6a3b48f940a2)

Showcase of high-quality samples selected by the active expansion strategy.
---
![sample](https://github.com/user-attachments/assets/eeb0b5a2-5ddc-44a5-a18d-66b779ccf739)

Usage
---
1. Environment
```
Linux with Python ≥ 3.8
conda create -n PENet python=3.11.5
conda activate PENet
torch==1.11.0
cuda==11.3
opencv-python==4.9.0.80
```
2. Test

Download our pre-trained models and prepare the test datasets. [Baidu Pan link:](https://pan.baidu.com/s/1_T8b9eCjVE0oaCvD_jRhJw)    code: 0825
```
python test.py
```

Citation
===
```
@article{wang2024progressive,
  title={Progressive Expansion for Semi-supervised Bi-modal Salient Object Detection},
  author={Wang, Jie and Zhang, Zihao and Yu, Nana and Han, Yahong},
  journal={Pattern Recognition},
  year={2024},
  publisher={Elsevier}
}
```


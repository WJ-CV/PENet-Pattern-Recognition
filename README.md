# PENet：Progressive Expansion for Semi-supervised Bi-modal Salient Object Detection
The paper has been accepted for publication in the journal Pattern Recognition.

Abstract
---
Existing bi-modal salient object detection (SOD) methods primarily rely on fully supervised training strategies that require extensive manual annotation. Undoubtedly, extensive manual annotation is time-consuming and laborious, and the fully supervised strategy is also prone to overfitting on the training set. Therefore, we introduce a semi-supervised learning architecture (SSLA) to alleviate these problems while ensuring detection performance. Considering that the inherent training mode and concise architecture of basic SSLA will limit its ability to effectively explore the learning potential of the model, we further propose two optimization strategies, dynamic adjustment and active expansion. Specifically, we dynamically adjust the supervision scheme for unlabeled samples during training so that the model can continuously utilize the model's gains (pseudo labels) to supervise and guide the model to further explore the unlabeled samples. Furthermore, the active expansion strategy enables the model to acquire more beneficial supervised information and focuses its attention on difficult-to-segment samples. In summary, an effective progressive expansion network (PENet) architecture for semi-supervised bi-modal SOD is proposed. Extensive experiments indicate that our PENet architecture, while effectively alleviating 90% of annotation burdens, has achieved highly competitive results in RGB-T and RGB-D tasks compared to fully supervised methods. The performance is even more pronounced during cross-dataset testing.

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

Download the ckeckpoints of our model from [BaiduYun](https://pan.baidu.com/s/1_T8b9eCjVE0oaCvD_jRhJw) (fetch code: 0825) and prepare the test datasets.
```
python test.py
```
3. Evalutation

We use the widely adopted Matlab-based saliency evaluation toolbox to generate metrics. [Evalutation toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox)


Citation
===
```
@article{wang2024progressive,
  title={Progressive expansion for semi-supervised Bi-modal salient object detection},
  author={Wang, Jie and Zhang, Zihao and Yu, Nana and Han, Yahong},
  journal={Pattern Recognition},
  pages={110868},
  year={2024},
  publisher={Elsevier}
}
```


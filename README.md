
# History

## [CVPR-2025-reading-papers-with-code](https://github.com/Paper2Chinese/CVPR-2025-reading-papers-with-code/blob/main/CVPR-2025-reading-papers-with-code.md) 

# CVPR-2026-reading-papers-with-code 

## 收集CVPR 2026论文&源码
## 收集全网对CVPR 2026论文的优质讲解

---

> 注1：欢迎各位作者大佬提交issue，分享CVPR 2026论文和开源项目！
>
> 注2：关于CV领域顶级期刊（TPAMI、IJCV等）论文解读大盘点，详见： [https://github.com/Paper2Chinese/Paper2Chinese](https://github.com/Paper2Chinese/Paper2Chinese)
>
> 注3：关于人工智能领域**NeurIPS顶会**论文解读大盘点，详见： [https://github.com/Paper2Chinese/NeurIPS2024-Reading-Paper-With-Code](https://github.com/Paper2Chinese/NeurIPS2024-Reading-Paper-With-Code)



# 【CVPR 2026 论文开源目录】


| [3DGS(Gaussian Splatting)](#3DGS) | [Mamba / (SSM)](#Mamba) | [Avatars](#Avatars) | [Backbone](#Backbone) | [CLIP](#CLIP) | [MAE](#MAE) |[联邦学习(Federated Learning)](#FL) |  
|-------|-------|-------| --------|--------|--------|--------|
| [多模态大语言模型(MLLM)](#MLLM) | [大语言模型(LLM)](#LLM) | [视觉语言模型(VLM)](#VLM) | [多模态(Multi-modal)](#multimodal)  | [NAS](#NAS)   |  [OCR](#OCR)  |  [NeRF](#NeRF)  |   
| [视觉问答(Visual Question Answering)](#VQA) | [强化学习(Reinforcement Learning)](#RL) | [扩散模型(Diffusion Models)](#Diffusion) |  [ReID(重识别)](#ReID) |  [长尾分布(Long-Tail)](#Long-Tail) | [视频压缩(Video Compression)](#VC) |   |   
|[增量学习(Incremental Learning)](#IL) |[数据增强(Data Augmentation)](#DA) | [目标检测(Object Detection)](#Object-Detection)|[异常检测(Anomaly Detection)](#Anomaly-Detection) | [目标跟踪(Visual Tracking)](#VT)|[语义分割(Semantic Segmentation)](#Semantic-Segmentation) | [实例分割(Instance Segmentation)](#Instance-Segmentation)| 
|[医学图像(Medical Image)](#MI) |[医学图像分割(Medical Image Segmentation)](#MIS) |[视频目标分割(Video Object Segmentation)](#VOS) |[视频实例分割(Video Instance Segmentation)](#VIS) | [参考图像分割(Referring Image Segmentation)](#RIS) |  [图像抠图(Image Matting)](#Matting)| [图像编辑(Image Editing)](#Image-Editing)|
|[具身智能](Embodied-AI)|[Prompt](#Prompt) | [自监督学习(Self-supervised Learning)](#SSL)   |  [生物工程(bioengineering)](#bio)| [Low-level Vision](#LLV)|[超分辨率(Super-Resolution)](#SR) |[去模糊(Deblur)](#Deblur)|
|[生成对抗网络(GAN)](#GAN) |[3D点云(3D Point Cloud)](#3D-Point-Cloud) |[3D目标检测(3D Object Detection)](#3DOD) | [3D语义分割(3D Semantic Segmentation)](#3DSS)|[3D目标跟踪(3D Object Tracking)](#3D-Object-Tracking) |[3D语义场景补全(3D Semantic Scene Completion)](#3DSSC) |[视频理解(Video Understanding)](#Video-Understanding)|
|[3D人体姿态估计(3D Human Pose Estimation)](#3D-Human-Pose-Estimation) |[3D人体Mesh估计(3D Human Mesh Estimation)](#3D-Human-Pose-Estimation) | [少样本学习(Few-Shot Learning)](#FewShot)| [图像生成(Image Generation)](#Image-Generation)|[视频生成(Video Generation)](#Video-Generation) |[3D生成(3D Generation)](#3D-Generation) | [图像压缩(Image Compression)](#IC)|
|[持续学习(Continual Learning)](#CL) |[行为识别(Action Recognition)](#Action-Recognition) | [行为检测(Action Detection)](#Action-Detection)|[人脸识别(Face Recognition)](#face-recognition) |[文本检测(Text Detection)](#Text-Detection) | [知识蒸馏(Knowledge Distillation)](#KD)|[三维重建(3D Reconstruction)](#3D-Reconstruction)
| [GNN](#GNN) | [DETR](#DETR)  |  [Vision Transformer](#Vision-Transformer) |[全景分割(Panoptic Segmentation)](#Panoptic-Segmentation)| [去噪(Denoising)](#Denoising) |[自动驾驶(Autonomous Driving)](#Autonomous-Driving)| [3D配准(3D Registration)](#3D-Registration) | 
| [模型剪枝(Model Pruning)](#Pruning) |[深度估计(Depth Estimation)](#Depth-Estimation) |[轨迹预测(Trajectory Prediction)](#TP) |[车道线检测(Lane Detection)](#Lane-Detection) |[图像描述(Image Captioning)](#Image-Captioning) | [手语识别(Sign Language Recognition)](#SLR)|[视频预测(Video Prediction)](#Video-Prediction)  | 
|[新视点合成(Novel View Synthesis)](#NVS) |[Zero-Shot Learning(零样本学习)](#ZSL) |[立体匹配(Stereo Matching)](#Stereo-Matching) | [特征匹配(Feature Matching)](#Feature-Matching)| [场景图生成(Scene Graph Generation)](#SGG) |  [计数(Counting)](#Counting)|[隐式神经表示(Implicit Neural Representations)](#INR) | 
|[图像质量评价(Image Quality Assessment)](#IQA) |[视频质量评价(Video Quality Assessment)](#Video-Quality-Assessment) |[数据集(Datasets)](#Datasets) |[反学习(Machine Unlearning)](#Unlearning) |[新任务(New Tasks)](#New-Tasks) |[模型加速(Improving Reasoning)](#Improving-Reasoning) |[时间序列(Time Series)](#Time-Series) | 
|[其他(Others)](#Others) |[脉冲网络](#SNN) |[图像检索](#IRetrieval) | [图像去雾(Dehazing)](#Dehazing) | | | | 

<a name="Dehazing"></a>
# 图像去雾(Dehazing)



<a name="EmAI"></a>
# 具身智能（Embodied AI）

#### GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer
- Link：https://arxiv.org/abs/2602.20871
- Code：

#### Probing and Bridging Geometry-Interaction Cues for Affordance Reasoning in Vision Foundation Models
- Link：https://arxiv.org/abs/2602.20501
- Code：https://github.com/facebookresearch/affordance-probing

#### MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation
- Link：https://arxiv.org/abs/2511.10376
- Code：

#### MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents
- Link：https://arxiv.org/abs/2602.20412
- Code：

<a name="3DGS"></a>

# 3DGS(Gaussian Splatting)
#### BrepGaussian: CAD reconstruction from Multi-View Images with Gaussian Splatting
- Link：[https://arxiv.org/abs/2602.21105](https://arxiv.org/abs/2602.21105)
- Code：

#### Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting
- Link：[https://arxiv.org/abs/2602.20933](https://arxiv.org/abs/2602.20933)
- Code：[https://sk-fun.fun/DropAnSH-GS](https://sk-fun.fun/DropAnSH-GS)

#### RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing
- Link：https://arxiv.org/abs/2602.19753
- Code：https://github.com/yyyykf/RAP

<a name="3D-Reconstruction"></a>
# 三维重建(3D Reconstruction)

#### Global-Aware Edge Prioritization for Pose Graph Initialization
- Link：[https://arxiv.org/abs/2602.21963](https://arxiv.org/abs/2602.21963)
- Code：[https://github.com/weitong8591/global_edge_prior](https://github.com/weitong8591/global_edge_prior)

#### MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second
- Link：https://arxiv.org/abs/2507.10065
- Code：

#### tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction
- Link：https://arxiv.org/abs/2602.20089
- Code：

<a name="Pruning"></a>
# 模型剪枝(Model Pruning)

#### Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers
- Link：https://arxiv.org/abs/2511.16156
- Code：https://github.com/OPPO-Mente-Lab/Qwen-Image-Pruning

<a name="Depth-Estimation"></a>
# 深度估计(Depth Estimation)

<a name="TP"></a>
# 轨迹预测(Trajectory Prediction)



<a name="Mamba"></a>

# Mamba / SSM



<a name="Avatars"></a>

# Avatars





<a name="Autonomous-Driving"></a>
# 自动驾驶

#### Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos
- Link：[https://arxiv.org/abs/2602.22091](https://arxiv.org/abs/2602.22091)
- Code：

#### LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration
- Link：[https://arxiv.org/abs/2602.21754](https://arxiv.org/abs/2602.21754)
- Code：

#### RAYNOVA: Scale-Temporal Autoregressive World Modeling in Ray Space
- Link：[https://arxiv.org/abs/2602.20685](https://arxiv.org/abs/2602.20685)
- Code：[https://raynova-ai.github.io/](https://raynova-ai.github.io/)


#### HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles
- Link：[https://arxiv.org/abs/2602.21333](https://arxiv.org/abs/2602.21333)
- Code：[https://horizonforge.github.io/](https://horizonforge.github.io/)

#### NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning
- Link：[https://arxiv.org/abs/2602.21172](https://arxiv.org/abs/2602.21172)
- Code：

#### Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule
- Link：https://arxiv.org/abs/2506.09217
- Code：https://github.com/datadrivenwheels/PCD_Python

#### SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World
- Link：https://arxiv.org/abs/2602.18887
- Code：

<a name="Backbone"></a>
# Backbone
#### Hyperbolic Busemann Neural Networks
- Link：[https://arxiv.org/abs/2602.18858](https://arxiv.org/abs/2602.18858)
- Code：[https://github.com/GitZH-Chen/HBNN](https://github.com/GitZH-Chen/HBNN)

#### PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning
- Link：https://arxiv.org/abs/2602.20537
- Code：https://github.com/fhjdqaq/PFGNet


<a name="CLIP"></a>
# CLIP
#### CLIPoint3D: Language-Grounded Few-Shot Unsupervised 3D Point Cloud Domain Adaptation
- Link：https://arxiv.org/abs/2602.20160
- Code：https://github.com/SarthakM320/CLIPoint3D


<a name="MAE"></a>
# MAE
#### Detecting AI-Generated Forgeries via Iterative Manifold Deviation Amplification
- Link：https://arxiv.org/abs/2602.18842
- Code：


<a name="OCR"></a>
# OCR
#### TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering
- Link：[https://arxiv.org/abs/2602.20903](https://arxiv.org/abs/2602.20903)
- Code：[https://github.com/CIawevy/TextPecker](https://github.com/CIawevy/TextPecker)


<a name="Occupancy"></a>

# Occupancy




<a name="NeRF"></a>
# NeRF



<a name="DETR"></a>
# DETR

#### EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer
- Link：[https://arxiv.org/abs/2602.20985](https://arxiv.org/abs/2602.20985)
- Code：


<a name="GNN"></a>
# GNN


<a name="Prompt"></a>
# Prompt

#### Towards Calibrating Prompt Tuning of Vision-Language Models
- Link：https://arxiv.org/abs/2602.19024
- Code：


<a name="LLM"></a>
# 大语言模型(LLM)

#### VecGlypher: Unified Vector Glyph Generation with Language Models
- Link：[https://arxiv.org/abs/2602.21461](https://arxiv.org/abs/2602.21461)
- Code：[https://xk-huang.github.io/VecGlypher/](https://xk-huang.github.io/VecGlypher/)


<a name="VLM"></a>
# 视觉语言模型(LLM)

#### Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness
- Link：https://arxiv.org/abs/2602.19615
- Code：

<a name="MLLM"></a>
# 多模态大语言模型(MLLM)

#### ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding
- Link：https://arxiv.org/abs/2602.16412
- Code：


#### WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs
- Link：[https://arxiv.org/abs/2602.22142](https://arxiv.org/abs/2602.22142)
- Code：[https://zhangyl4.github.io/publications/weavetime/](https://zhangyl4.github.io/publications/weavetime/)

#### MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping
- Link：https://arxiv.org/abs/2511.15690
- Code：https://github.com/ModelTC/MoDES

#### Echoes of Ownership: Adversarial-Guided Dual Injection for Copyright Protection in MLLMs
- Link：https://arxiv.org/abs/2602.18845
- Code：

<a name="multimodal"></a>
# 多模态

#### MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning
- Link：[https://arxiv.org/abs/2602.20223](https://arxiv.org/abs/2602.20223)
- Code：[https://github.com/too-z/MultiModalPFN](https://github.com/too-z/MultiModalPFN)

#### Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery
- Link：https://arxiv.org/abs/2602.19910
- Code：

#### CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning
- Link：https://arxiv.org/abs/2602.19605
- Code：

#### Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis
- Link：https://arxiv.org/abs/2602.19585
- Code：

#### CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion
- Link：https://arxiv.org/abs/2602.19140
- Code：

<a name="NAS"></a>
# NAS

<a name="VQA"></a>
## 视觉问答(Visual Question Answering)

#### SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models
- Link：[https://arxiv.org/abs/2602.20901](https://arxiv.org/abs/2602.20901)
- Code：[https://github.com/xieyc99/SpatiaLQA](https://github.com/xieyc99/SpatiaLQA)

<a name="RL"></a>
## 强化学习(Reinforcement Learning) 




<a name="ReID"></a>
# ReID(重识别)




<a name="Long-Tail"></a>
# 长尾分布(Long-Tail)
#### Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces
- Link：https://arxiv.org/abs/2602.20068
- Code：

<a name="VC"></a>
# 视频压缩(Video Compression)


<a name="Diffusion"></a>
# 扩散模型(Diffusion Models)

#### ConceptPrism: Concept Disentanglement in Personalized Diffusion Models via Residual Token Optimization
- Link：https://arxiv.org/abs/2602.19575
- Code：

#### SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models
- Link：https://arxiv.org/abs/2602.18993
- Code：


<a name="Vision-Transformer"></a>
# Vision Transformer



<a name="Panoptic-Segmentation"></a>
# 全景分割(Panoptic Segmentation)




<a name="VL"></a>
# 视觉和语言(Vision-Language)

#### StructXLIP: Enhancing Vision-language Models with Multimodal Structural Cues
- Link：https://arxiv.org/abs/2503.07853
- Code：https://github.com/intelligolabs/StructXLIP

<a name="Object-Detection"></a>
# 目标检测(Object Detection)

#### Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection
- Link：https://arxiv.org/abs/2512.17514
- Code：

<a name="DA"></a>
## 数据增强(Data Augmentation)




<a name="Anomaly-Detection"></a>
# 异常检测(Anomaly Detection)

#### The Invisible Gorilla Effect in Out-of-distribution Detection
- Link：https://arxiv.org/abs/2602.19944
- Code：https://github.com/HarryAnthony/Invisible_Gorilla_Effect

#### GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning
- Link：[https://arxiv.org/abs/2602.19206](https://arxiv.org/abs/2602.19206)
- Code：[https://github.com/zhushengxinyue/GS-CLIP](https://github.com/zhushengxinyue/GS-CLIP)

#### No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection
- Link：https://arxiv.org/abs/2602.19248
- Code：https://github.com/VitaminCreed/LAVIDA

#### SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images
- Link：https://arxiv.org/abs/2602.20409
- Code：

<a name="VT"></a>
# 目标跟踪(Object Tracking)

#### Changes in Real Time: Online Scene Change Detection with Multi-View Fusion
- Link：https://arxiv.org/abs/2511.12370
- Code：https://chumsy0725.github.io/O-SCD/

#### Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction
- Link：https://arxiv.org/abs/2602.18996
- Code：https://github.com/shannany0606/CCMP


<a name="Semantic-Segmentation"></a>
# 语义分割(Semantic Segmentation)

<a name="Instance-Segmentation"></a>
# 实例分割(Instance Segmentation)


<a name="FewShot"></a>
# 少样本学习(Few-Shot Learning)

#### MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification
- Link：[https://arxiv.org/abs/2602.20873](https://arxiv.org/abs/2602.20873)
- Code：[https://github.com/JiahaoXu-god/CVPR2026_MUSE](https://github.com/JiahaoXu-god/CVPR2026_MUSE)

#### Learning Multi-Modal Prototypes for Cross-Domain Few-Shot Object Detection
- Link：https://arxiv.org/abs/2602.18811
- Code：
  
<a name="bio"></a>
# 生物医学


<a name="MI"></a>
# 医学图像(Medical Image)

#### MRI Contrast Enhancement Kinetics World Model
- Link：https://arxiv.org/abs/2602.19285
- Code：https://github.com/DD0922/MRI-Contrast-Enhancement-Kinetics-World-Model

#### Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning
- Link：https://arxiv.org/abs/2602.18867
- Code：

<a name="MIS"></a>
# 医学图像分割(Medical Image Segmentation)




<a name="VOS"></a>
# 视频目标分割(Video Object Segmentation)



<a name="Action-Detection"></a>
# 行为检测(Action Detection)

<a name="face-recognition"></a>

# 人脸识别(Face Recognition)

#### RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces
- Link：https://arxiv.org/abs/2602.20618
- Code：

#### IDperturb: Enhancing Variation in Synthetic Face Generation via Angular Perturbation
- Link：https://arxiv.org/abs/2602.18831
- Code：


<a name="3D-Point-Cloud"></a>
# 3D点云(3D-Point-Cloud)
#### Universal 3D Shape Matching via Coarse-to-Fine Language Guidance
- Link：https://arxiv.org/abs/2511.23055
- Code：


<a name="SSL"></a>
# 自监督学习(Self-supervised Learning)



<a name="bio"></a>
# 生物工程(bioengineering)




<a name="FL"></a>
# 联邦学习(Federated Learning)

#### FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning
- Link：[https://arxiv.org/abs/2602.21399](https://arxiv.org/abs/2602.21399)
- Code：[https://github.com/alinadevkota/FedVG](https://github.com/alinadevkota/FedVG)

<a name="IL"></a>
# 增量学习(Incremental Learning)



<a name="#3DOD"></a>
# 3D目标检测(3D Object Detection)

#### Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception
- Link：https://arxiv.org/abs/2602.19596
- Code：https://github.com/yihangtao/MVIG.git

#### SABER: Spatially Consistent 3D Universal Adversarial Objects for BEV Detectors
- Link：https://arxiv.org/abs/2505.22499
- Code：



<a name="3DOD"></a>
# 3D语义分割(3D Semantic Segmentation)




<a name="Image-Editing"></a>
# 图像编辑(Image Editing)

#### Cycle-Consistent Tuning for Layered Image Decomposition
- Link：[https://arxiv.org/abs/2602.20989](https://arxiv.org/abs/2602.20989)
- Code：[https://vcc.tech/research/2026/ImgDecom](https://vcc.tech/research/2026/ImgDecom)

#### ChordEdit: One-Step Low-Energy Transport for Image Editing
- Link：https://arxiv.org/abs/2602.19083
- Code：

<a name="Image-Inpainting"></a>
# 图像补全/图像修复(Image Inpainting)


<a name="GAN"></a>
# 生成对抗网络(GAN)




<a name="Video-Editing"></a>
# 视频编辑(Video Editing)

#### Object-WIPER: Training-Free Object and Associated Effect Removal in Videos
- Link：https://arxiv.org/abs/2601.06391
- Code：

<a name="LLV"></a>
# Low-level Vision

#### Lumosaic: Hyperspectral Video via Active Illumination and Coded-Exposure Pixels
- Link：[https://arxiv.org/abs/2602.22140](https://arxiv.org/abs/2602.22140)
- Code：

#### MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision
- Link：[https://arxiv.org/abs/2602.20689](https://arxiv.org/abs/2602.20689)
- Code：[https://cvpr26-matched.github.io](https://cvpr26-matched.github.io)


<a name="SR"></a>
# 超分辨率(Super-Resolution)





<a name="Denoising"></a>
# 去噪(Denoising)

<a name="3D-Human-Pose-Estimation"></a>



<a name="Image-Generation"></a>

# 图像生成(Image Generation)



<a name="Video-Generation"></a>
# 视频生成(Video Generation)
#### Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context
- Link：[https://arxiv.org/abs/2602.21929](https://arxiv.org/abs/2602.21929)
- Code：

#### ExpPortrait: Expressive Portrait Generation via Personalized Representation
- Link：https://arxiv.org/abs/2602.19900
- Code：

#### LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation
- Link：https://arxiv.org/abs/2510.08318
- Code：



#### Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models
- Link：[https://arxiv.org/abs/2602.20981](https://arxiv.org/abs/2602.20981)
- Code：

#### The devil is in the details: Enhancing Video Virtual Try-On via Keyframe-Driven Details Injection
- Link：https://arxiv.org/abs/2512.20340
- Code：

<a name="3D-Generation"></a>
# 3D生成

#### Easy3E: Feed-Forward 3D Asset Editing via Rectified Voxel Flow
- Link：[https://arxiv.org/abs/2602.21499](https://arxiv.org/abs/2602.21499)
- Code：

#### BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation
- Link：https://arxiv.org/abs/2602.18873
- Code：

<a name="Video-Understanding"></a>
# 视频理解(Video Understanding)
#### LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding
- Link：[https://arxiv.org/abs/2602.20913](https://arxiv.org/abs/2602.20913)
- Code：[https://github.com/qiujihao19/LongVideo-R1](https://github.com/qiujihao19/LongVideo-R1)

#### Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding
- Link：https://arxiv.org/abs/2602.18977
- Code：https://github.com/th-nesh/Frame2Freq


<a name="3D-Human-Pose-Estimation"></a>
# 3D人体姿态估计(3D Human Pose Estimation)

#### SIMSPINE: A Biomechanics-Aware Simulation Framework for 3D Spine Motion Annotation and Benchmarking
- Link：[https://arxiv.org/abs/2602.20792](https://arxiv.org/abs/2602.20792)
- Code：

#### VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery
- Link：https://arxiv.org/abs/2602.19180
- Code：

<a name="CL"></a>
# 持续学习(Continual Learning)




<a name="Action-Recognition"></a>
# 行为识别(Action Recognition)

#### BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment
- Link：https://arxiv.org/abs/2602.19170
- Code：


<a name="KD"></a>
# 知识蒸馏(Knowledge Distillation)
#### Momentum Memory for Knowledge Distillation in Computational Pathology
- Link：[https://arxiv.org/abs/2602.21395](https://arxiv.org/abs/2602.21395)
- Code：

#### Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation
- Link：[https://arxiv.org/abs/2602.19863](https://arxiv.org/abs/2602.19863)
- Code：[https://wolfilip.github.io/DEO/](https://wolfilip.github.io/DEO/)

<a name="IC"></a>
# 图像压缩(Image Compression)



<a name="ZSL"></a>
# Zero-Shot Learning(零样本学习)



<a name="Stereo-Matching"></a>
# 立体匹配(Stereo Matching)

#### Pip-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching
- Link：https://arxiv.org/abs/2602.19112
- Code：https://github.com/XPENG-Aridge-AI


<a name="SGG"></a>
# 场景图生成(Scene Graph Generation)



<a name="Counting"></a>
# 计数(Counting)



<a name="INR"></a>
# 隐式神经表示(Implicit Neural Representations)




<a name="IQA"></a>
# 图像质量评价(Image Quality Assessment)

#### How to Take a Memorable Picture? Empowering Users with Actionable Feedback
- Link：[https://arxiv.org/abs/2602.21877](https://arxiv.org/abs/2602.21877)
- Code：[https://laitifranz.github.io/MemCoach/](https://laitifranz.github.io/MemCoach/)

<a name="Video-Quality-Assessment"></a>
# 视频质量评价(Video Quality Assessment)



<a name="Datasets"></a>
# 数据集(Datasets)
#### E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought
- Link：[https://arxiv.org/abs/2602.21698](https://arxiv.org/abs/2602.21698)
- Code：[https://github.com/4mm7/E-comIQ-ZH](https://github.com/4mm7/E-comIQ-ZH)

<a name="Unlearning"></a>
# 反学习(Machine Unlearning)



<a name="New-Tasks"></a>
# 新任务(New Tasks)



<a name="Improving-Reasoning"></a>
# 模型加速(Improving Reasoning)

#### Variation-aware Vision Token Dropping for Faster Large Vision-Language Models
- Link：[https://arxiv.org/abs/2509.01552](https://arxiv.org/abs/2509.01552)
- Code：[https://github.com/xuyang-liu16/V2Drop](https://github.com/xuyang-liu16/V2Drop)

#### Model Merging in the Essential Subspace
- Link：https://arxiv.org/abs/2602.20208
- Code：



<a name="Time-Series"></a>
# 时间序列(Time Series)
#### STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting
- Link：https://arxiv.org/abs/2509.25210
- Code：


<a name="SNN"></a>

# 脉冲网络


<a name="IRetrieval"></a>
# 图像检索

#### TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking
- Link：https://arxiv.org/abs/2602.18863
- Code：

# 其他(Others)
#### NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training
- Link：https://arxiv.org/abs/2602.22059
- Code：

#### Defending Unauthorized Model Merging via Dual-Stage Weight Protection
- Link：https://arxiv.org/abs/2511.11851
- Code：





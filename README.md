
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

#### Bilevel Layer-Positioning LoRA for Real Image Dehazing
- Link：https://arxiv.org/abs/2603.10872
- Code：https://github.com/YanZhang-zy/BiLaLoRA

#### UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization
- Link：https://arxiv.org/abs/2603.03967
- Code：https://github.com/QianfengY/UniRain

<a name="EmAI"></a>
# 具身智能（Embodied AI）

#### MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation
- Link：https://arxiv.org/abs/2511.10376
- Code：https://github.com/ylwhxht/MSGNav

#### ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation
- Link：https://arxiv.org/abs/2603.15169
- Code：https://sites.google.com/view/force-vla2/home

#### RealVLG-R1: A Large-Scale Real-World Visual-Language Grounding Benchmark for Robotic Perception and Manipulation
- Link：https://arxiv.org/abs/2603.14880
- Code：https://github.com/lif314/RealVLG-R1

#### SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics
- Link：https://arxiv.org/abs/2603.12193
- Code：https://lmzpai.github.io/SaPaVe

#### Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation
- Link：https://arxiv.org/abs/2603.09506
- Code：

#### MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent
- Link：https://arxiv.org/abs/2511.18810
- Code：https://mergevla.github.io/

#### When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models
- Link：https://arxiv.org/abs/2511.21192
- Code：

#### Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation
- Link：https://arxiv.org/abs/2603.09506
- Code：https://github.com/AutoCompSysLab/ContextNav

#### Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation
- Link：https://arxiv.org/abs/2508.05186
- Code：https://github.com/HCPLab-SYSU/TAVP.git

#### Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning
- Link：https://arxiv.org/abs/2603.18495
- Code：



#### Structural Action Transformer for 3D Dexterous Manipulation
- Link：https://arxiv.org/abs/2603.03960
- Code：

#### Action-Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation
- Link：https://arxiv.org/abs/2602.23814
- Code：https://github.com/Chongyang-99/GAP.git

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

#### CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image
- Link：https://arxiv.org/abs/2603.17779
- Code：

#### ReLaGS: Relational Language Gaussian Splatting
- Link：https://arxiv.org/abs/2603.17605
- Code：https://dfki-av.github.io/ReLaGS/

#### Motion-Aware Animatable Gaussian Avatars Deblurring
- Link：https://arxiv.org/abs/2411.16758
- Code：https://github.com/MyNiuuu/MAD-Avatar

#### LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates
- Link：https://arxiv.org/abs/2510.09881
- Code：

#### EMGauss: Continuous Slice-to-3D Reconstruction via Dynamic Gaussian Modeling in Volume Electron Microscopy
- Link：https://arxiv.org/abs/2512.06684
- Code：https://raynehe.github.io/EMGauss/

#### PhyGaP: Physically-Grounded Gaussians with Polarization Cues
- Link：https://arxiv.org/abs/2603.14001
- Code：

#### Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists
- Link：https://arxiv.org/abs/2603.09277
- Code：https://github.com/MachinePerceptionLab/ShorterSplatting

#### REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting
- Link：https://arxiv.org/abs/2510.16410
- Code：https://ChangyueShi.github.io/REALM

#### VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM
- Link：https://arxiv.org/abs/2603.09673
- Code：

#### E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction
- Link：https://arxiv.org/abs/2603.14684
- Code：

#### ProgressiveAvatars: Progressive Animatable 3D Gaussian Avatars
- Link：https://arxiv.org/abs/2603.16447
- Code：https://ustc3dv.github.io/ProgressiveAvatars/

#### Let it Snow! Animating 3D Gaussian Scenes with Dynamic Weather Effects via Physics-Guided Score Distillation
- Link：https://arxiv.org/abs/2504.05296
- Code：https://galfiebelman.github.io/let-it-snow/

#### OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution
- Link：https://arxiv.org/abs/2603.02134
- Code：https://xiac20.github.io/OnlineX/

#### STAvatar: Soft Binding and Temporal Density Control for Monocular 3D Head Avatars Reconstruction
- Link：https://arxiv.org/abs/2511.19854
- Code：https://jiankuozhao.github.io/STAvatar/

#### Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting
- Link：[https://arxiv.org/abs/2602.20933](https://arxiv.org/abs/2602.20933)
- Code：[https://sk-fun.fun/DropAnSH-GS](https://sk-fun.fun/DropAnSH-GS)

#### RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing
- Link：https://arxiv.org/abs/2602.19753
- Code：https://github.com/yyyykf/RAP

<a name="3D-Reconstruction"></a>
# 三维重建(3D Reconstruction)
#### GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis
- Link：https://arxiv.org/abs/2603.01010
- Code：

#### Talking Together: Synthesizing Co-Located 3D Conversations from Audio
- Link：https://arxiv.org/abs/2603.08674
- Code：

#### Parallelised Differentiable Straightest Geodesics for 3D Meshes
- Link：https://arxiv.org/abs/2603.15780
- Code：https://circle-group.github.io/research/DSG

#### FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting
- Link：https://arxiv.org/abs/2602.24084
- Code：

#### A2Z-10M+: Geometric Deep Learning with A-to-Z BRep Annotations for AI-Assisted CAD Modeling and Reverse Engineering
- Link：https://arxiv.org/abs/2603.12605
- Code：

#### Order Matters: 3D Shape Generation from Sequential VR Sketches
- Link：https://arxiv.org/abs/2512.04761
- Code：https://chenyizi086.github.io/VRSketch2Shape_website

#### CustomTex: High-fidelity Indoor Scene Texturing via Multi-Reference Customization
- Link：https://arxiv.org/abs/2603.19121
- Code：

#### PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery
- Link：https://arxiv.org/abs/2603.17571
- Code：



#### SimRecon: SimReady Compositional Scene Reconstruction from Real Videos
- Link：https://arxiv.org/abs/2603.02133
- Code：https://xiac20.github.io/SimRecon/

#### MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer
- Link：https://arxiv.org/abs/2603.05078
- Code：https://hellexf.github.io/MoRe/

#### RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations
- Link：https://arxiv.org/abs/2603.01194
- Code：https://npucvr.github.io/RnG

#### tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction
- Link：https://arxiv.org/abs/2602.20160
- Code：https://cwchenwang.github.io/tttLRM

#### FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting
- Link：https://arxiv.org/abs/2602.24084
- Code：

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

#### When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs
- Link：https://arxiv.org/abs/2512.07580
- Code：https://github.com/YahongWang1/Information-Horizon

#### Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving
- Link：https://arxiv.org/abs/2508.13305
- Code：https://github.com/MinhaoXiong/Prune2Drive.git

#### Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers
- Link：https://arxiv.org/abs/2511.16156
- Code：https://github.com/OPPO-Mente-Lab/Qwen-Image-Pruning

<a name="Depth-Estimation"></a>
# 深度估计(Depth Estimation)




#### SpiderCam: Low-Power Snapshot Depth from Differential Defocus
- Link：https://arxiv.org/abs/2603.17910
- Code：

<a name="TP"></a>
# 轨迹预测(Trajectory Prediction)

#### FoSS: Modeling Long Range Dependencies and Multimodal Uncertainty in Trajectory Prediction via Fourier State Space Integration
- Link：https://arxiv.org/abs/2603.01284
- Code：

#### Recover to Predict: Progressive Retrospective Learning for Variable-Length Trajectory Prediction
- Link：https://arxiv.org/abs/2603.10597
- Code：https://github.com/zhouhao94/PRF

<a name="Mamba"></a>

# Mamba / SSM

#### DA-Mamba: Learning Domain-Aware State Space Model for Global-Local Alignment in Domain Adaptive Object Detection
- Link：https://arxiv.org/abs/2603.18757
- Code：

<a name="Avatars"></a>

# Avatars





<a name="Autonomous-Driving"></a>
# 自动驾驶

#### AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception
- Link：https://arxiv.org/abs/2603.17979
- Code：

#### CausalVAD: De-confounding End-to-End Autonomous Driving via Causal Intervention
- Link：https://arxiv.org/abs/2603.18561
- Code：

#### SimScale: Learning to Drive via Real-World Simulation at Scale
- Link：https://arxiv.org/abs/2511.23369
- Code：https://github.com/OpenDriveLab/SimScale

#### VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation
- Link：https://arxiv.org/abs/2603.12918
- Code：

#### All Vehicles Can Lie: Efficient Adversarial Defense in Fully Untrusted-Vehicle Collaborative Perception
- Link：https://arxiv.org/abs/2603.08498
- Code：

#### HG-Lane: High-Fidelity Generation of Lane Scenes under Adverse Weather and Lighting Conditions without Re-annotation
- Link：https://arxiv.org/abs/2603.10128
- Code：https://github.com/zdc233/HG-Lane

#### KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System
- Link：https://arxiv.org/abs/2512.20299
- Code：

#### SABER: Spatially Consistent 3D Universal Adversarial Objects for BEV Detectors
- Link：https://arxiv.org/abs/2505.22499
- Code：

#### Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos
- Link：https://arxiv.org/abs/2602.22091
- Code：

#### DriverGaze360: OmniDirectional Driver Attention with Object-Level Guidance
- Link：https://arxiv.org/abs/2512.14266
- Code：https://dfki-av.github.io/drivergaze360

#### CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion
- Link：https://arxiv.org/abs/2603.00682
- Code：https://github.com/CatOneTwo/CoLC

#### Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos
- Link：[https://arxiv.org/abs/2602.22091](https://arxiv.org/abs/2602.22091)
- Code：

#### Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving
- Link：https://arxiv.org/abs/2603.01007
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

#### FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Alignment
- Link：https://arxiv.org/abs/2505.11192
- Code：

#### Reevaluating the Intra-Modal Misalignment Hypothesis in CLIP
- Link：https://arxiv.org/abs/2603.16100
- Code：

#### CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection
- Link：https://arxiv.org/abs/2511.18519
- Code：






#### CLIPoint3D: Language-Grounded Few-Shot Unsupervised 3D Point Cloud Domain Adaptation
- Link：https://arxiv.org/abs/2602.20160
- Code：https://github.com/SarthakM320/CLIPoint3D

#### FluoCLIP: Stain-Aware Focus Quality Assessment in Fluorescence Microscopy
- Link：https://arxiv.org/abs/2602.23791
- Code：


<a name="MAE"></a>
# MAE
#### Detecting AI-Generated Forgeries via Iterative Manifold Deviation Amplification
- Link：https://arxiv.org/abs/2602.18842
- Code：

#### SARMAE: Masked Autoencoder for SAR Representation Learning
- Link：https://arxiv.org/abs/2512.16635
- Code：https://github.com/MiliLab/SARMAE

<a name="OCR"></a>
# OCR

#### What Is Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution
- Link：https://arxiv.org/abs/2602.06450
- Code：https://github.com/YesianRohn/UnionST


#### TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering
- Link：[https://arxiv.org/abs/2602.20903](https://arxiv.org/abs/2602.20903)
- Code：[https://github.com/CIawevy/TextPecker](https://github.com/CIawevy/TextPecker)

#### Efficient Document Parsing via Parallel Token Prediction
- Link：https://arxiv.org/abs/2603.15206
- Code：

#### D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping
- Link：https://arxiv.org/abs/2507.08492
- Code：https://github.com/xiaomore/D2Dewarp

<a name="Occupancy"></a>

# Occupancy

#### OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera
- Link：https://arxiv.org/abs/2511.03571
- Code：https://github.com/MasterHow/OneOcc

#### Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving
- Link：https://arxiv.org/abs/2603.01007
- Code：


<a name="NeRF"></a>
# NeRF

#### Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis
- Link：https://arxiv.org/abs/2603.12903
- Code：

#### Node-RF: Learning Generalized Continuous Space-Time Scene Dynamics with Neural ODE-based NeRFs
- Link：https://arxiv.org/abs/2603.12078
- Code：

#### Seeing through Light and Darkness: Sensor-Physics Grounded Deblurring HDR NeRF from Single-Exposure Images and Events
- Link：https://arxiv.org/abs/2601.15475
- Code：https://icvteam.github.io/See-NeRF.html

#### NERFIFY: A Multi-Agent Framework for Turning NeRF Papers into Code
- Link：https://arxiv.org/abs/2603.00805
- Code：

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

#### PHAC: Promptable Human Amodal Completion
- Link：https://arxiv.org/abs/2603.14741
- Code：

#### FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation
- Link：https://arxiv.org/abs/2603.04733
- Code：

#### FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation
- Link：https://arxiv.org/abs/2603.04733
- Code：

<a name="LLM"></a>
# 大语言模型(LLM)

#### VecGlypher: Unified Vector Glyph Generation with Language Models
- Link：[https://arxiv.org/abs/2602.21461](https://arxiv.org/abs/2602.21461)
- Code：[https://xk-huang.github.io/VecGlypher/](https://xk-huang.github.io/VecGlypher/)

#### LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference
- Link：https://arxiv.org/abs/2603.11605
- Code：https://jjkislele.github.io/LaMoGen/

<a name="VLM"></a>
# 视觉语言模型(LLM)

#### Draft and Refine with Visual Experts
- Link：https://arxiv.org/abs/2511.11005
- Code：https://github.com/EavnJeong/Draft-and-Refine-with-Visual-Experts

#### Interpretable Debiasing of Vision-Language Models for Social Fairness
- Link：https://arxiv.org/abs/2602.24014
- Code：

#### Ego: Embedding-Guided Personalization of Vision-Language Models
- Link：https://arxiv.org/abs/2603.09771
- Code：

#### GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training
- Link：https://arxiv.org/abs/2512.13043
- Code：

#### V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs
- Link：https://arxiv.org/abs/2511.20223
- Code：https://github.com/Summu77/V-Attack

#### It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand Spatial Reasoning in Vision-Language Models
- Link：https://arxiv.org/abs/2603.08011
- Code：

#### Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs
- Link：https://arxiv.org/abs/2603.02618
- Code：

#### AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition
- Link：https://arxiv.org/abs/2512.03794
- Code：https://github.com/AdaptVision/AdaptVision

#### DeAR: Fine-Grained VLM Adaptation by Decomposing Attention Head Roles
- Link：https://arxiv.org/abs/2603.01111
- Code：

#### Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks
- Link：https://arxiv.org/abs/2508.04097
- Code：https://ngoc-nguyen-0.github.io/SMI_AW/

#### Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness
- Link：https://arxiv.org/abs/2602.19615
- Code：

#### Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization
- Link：https://arxiv.org/abs/2602.24059
- Code：

<a name="MLLM"></a>
# 多模态大语言模型(MLLM)







#### No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection
- Link：https://arxiv.org/abs/2602.19248
- Code：https://github.com/VitaminCreed/LAVIDA

#### See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles
- Link：https://arxiv.org/abs/2509.13615
- Code：https://github.com/ZrW00/StaR

#### Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation
- Link：https://arxiv.org/abs/2509.22496
- Code：

#### Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients
- Link：https://arxiv.org/abs/2603.17809
- Code：https://github.com/ucas-xiang/QIG

#### How to Take a Memorable Picture? Empowering Users with Actionable Feedback
- Link：https://arxiv.org/abs/2602.21877
- Code：https://laitifranz.github.io/MemCoach/

#### Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token
- Link：https://arxiv.org/abs/2603.19026
- Code：https://github.com/ANDYZAQ/SELF1E

#### MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image Understanding
- Link：https://arxiv.org/abs/2512.02906
- Code：https://github.com/yf0412/MRD

#### OddGridBench: Exposing the Lack of Fine-Grained Visual Discrepancy Sensitivity in Multimodal Large Language Models
- Link：https://arxiv.org/abs/2603.09326
- Code：https://wwwtttjjj.github.io/OddGridBench/

#### FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction
- Link：https://arxiv.org/abs/2509.21029
- Code：

#### HIFICL: High-Fidelity In-Context Learning for Multimodal Tasks
- Link：https://arxiv.org/abs/2603.12760
- Code：https://github.com/bbbandari/HiFICL

#### Parallel In-context Learning for Large Vision Language Models
- Link：https://arxiv.org/abs/2603.16092
- Code：https://github.com/yshinya6/parallel-icl

#### Multi-Crit: Benchmarking Multimodal Judges on Pluralistic Criteria-Following
- Link：https://arxiv.org/abs/2511.21662
- Code：

#### SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs
- Link：https://arxiv.org/abs/2603.12382
- Code：https://risys-lab.github.io/SPARROW；https://github.com/RISys-Lab/SPARROW

#### Tokenization Allows Multimodal Large Language Models to Understand, Generate and Edit Architectural Floor Plans
- Link：https://arxiv.org/abs/2603.11640
- Code：

#### WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation
- Link：https://arxiv.org/abs/2603.10703
- Code：https://sites.google.com/view/walkgpt-26/home

#### The Coherence Trap: When MLLM-Crafted Narratives Exploit Manipulated Visual Contexts
- Link：https://arxiv.org/abs/2505.17476
- Code：

#### KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing
- Link：https://arxiv.org/abs/2602.04268
- Code：

#### IMAIA: Interactive Maps AI Assistant for Travel Planning and Geo-Spatial Intelligence
- Link：https://arxiv.org/abs/2507.06993
- Code：

#### LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models
- Link：https://arxiv.org/abs/2509.25896
- Code：

#### Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory
- Link：https://arxiv.org/abs/2603.15800
- Code：https://echosafe-mllm.github.io

#### Locate-then-Sparsify: Attribution Guided Sparse Strategy for Visual Hallucination Mitigation
- Link：https://arxiv.org/abs/2603.16284
- Code：

#### GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents
- Link：https://arxiv.org/abs/2603.15039
- Code：

#### Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought
- Link：https://arxiv.org/abs/2507.07685
- Code：https://github.com/yshinya6/red/

#### ViRC: Enhancing Visual Interleaved Mathematical CoT with Reason Chunking
- Link：https://arxiv.org/abs/2512.14654
- Code：https://github.com/Leon-LihongWang/ViRC

#### See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles
- Link：https://arxiv.org/abs/2509.13615
- Code：https://github.com/ZrW00/StaR

#### Evolutionary Multimodal Reasoning via Hierarchical Semantic Representation for Intent Recognition
- Link：https://arxiv.org/abs/2603.03827
- Code：https://github.com/thuiar/HIER

#### Training High-Level Schedulers with Execution-Feedback Reinforcement Learning for Long-Horizon GUI Automation
- Link：https://arxiv.org/abs/2511.22235
- Code：https://github.com/hehehahi4/CES

#### Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs
- Link：https://arxiv.org/abs/2510.00507
- Code：

#### MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models
- Link：https://arxiv.org/abs/2603.04800
- Code：https://github.com/alibaba/EfficientAI

#### EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning in Multimodal Large Language Models
- Link：https://arxiv.org/abs/2602.23802
- Code：


#### ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding
- Link：https://arxiv.org/abs/2602.16412
- Code：

#### Venus: Benchmarking and Empowering Multimodal Large Language Models for Aesthetic Guidance and Cropping
- Link：https://arxiv.org/abs/2602.23980
- Code：https://github.com/PKU-ICST-MIPL/Venus_CVPR2026

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

#### ConsistCompose: Unified Multimodal Layout Control for Image Composition
- Link：https://arxiv.org/abs/2511.18333
- Code：

#### Mixture of States: Routing Token-Level Dynamics for Multimodal Generation
- Link：https://arxiv.org/abs/2511.12207
- Code：https://haozheliu-st.github.io/mos-homepage/

#### Intrinsic Concept Extraction Based on Compositional Interpretability
- Link：https://arxiv.org/abs/2603.11795
- Code：

#### ParTY: Part-Guidance for Expressive Text-to-Motion Synthesis
- Link：https://arxiv.org/abs/2603.09611
- Code：https://github.com/VisualScienceLab-KHU/ParTY

#### x²-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space
- Link：https://arxiv.org/abs/2603.16671
- Code：

#### EI: Early Intervention for Multimodal Imaging based Disease Recognition
- Link：https://arxiv.org/abs/2603.17514
- Code：




















#### Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation
- Link：https://arxiv.org/abs/2603.00574
- Code：

#### Linking Modality Isolation in Heterogeneous Collaborative Perception
- Link：https://arxiv.org/abs/2603.00609
- Code：https://github.com/cxliu0314/CodeAlign

#### UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression
- Link：https://arxiv.org/abs/2509.25934
- Code：https://github.com/yuanzhao-CVLAB/UniMMAD

#### Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval
- Link：https://arxiv.org/abs/2603.01082
- Code：https://github.com/EIT-NLP/MCMR


#### Adaptive Confidence Regularization for Multimodal Failure Detection
- Link：https://arxiv.org/abs/2603.02200
- Code：https://github.com/mona4399/ACR

#### Cross-modal Identity Mapping: Minimizing Information Loss in Modality Conversion via Reinforcement Learning
- Link：https://arxiv.org/abs/2603.01696
- Code：

#### VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion
- Link：https://arxiv.org/abs/2503.23359
- Code：https://github.com/Linfeng-Tang/VideoFusion

#### U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation
- Link：https://arxiv.org/abs/2602.23739
- Code：

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

#### Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering
- Link：https://arxiv.org/abs/2603.13878
- Code：github.com/hahaha111111/Step-CoT

#### Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering
- Link：https://arxiv.org/abs/2603.12533
- Code：https://yuuraa.github.io/papers/choi2026egovqa

#### SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models
- Link：[https://arxiv.org/abs/2602.20901](https://arxiv.org/abs/2602.20901)
- Code：[https://github.com/xieyc99/SpatiaLQA](https://github.com/xieyc99/SpatiaLQA)

<a name="RL"></a>
## 强化学习(Reinforcement Learning) 


#### Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry
- Link：https://arxiv.org/abs/2511.21083
- Code：

#### Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning
- Link：https://arxiv.org/abs/2603.11346
- Code：https://yutoshibata07.github.io/AssistMimic-projectpage/

#### RL-ScanIQA: Reinforcement-Learned Scanpaths for Blind 360°Image Quality Assessment
- Link：https://arxiv.org/abs/2603.14297
- Code：https://github.com/wangyuji1/RLScanIQA.git

#### Specificity-aware reinforcement learning for fine-grained open-world classification
- Link：https://arxiv.org/abs/2603.03197
- Code：https://github.com/s-angheben/SpeciaRL

#### From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection
- Link：https://arxiv.org/abs/2602.20630
- Code：

#### OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation
- Link：https://arxiv.org/abs/2509.18600
- Code：


<a name="ReID"></a>
# ReID(重识别)




<a name="Long-Tail"></a>
# 长尾分布(Long-Tail)
#### Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces
- Link：https://arxiv.org/abs/2602.20068
- Code：

#### Meta-Learning Hyperparameters for Parameter Efficient Fine-Tuning
- Link：https://arxiv.org/abs/2603.01759
- Code：https://github.com/doem97/metalora

<a name="VC"></a>
# 视频压缩(Video Compression)


<a name="Diffusion"></a>
# 扩散模型(Diffusion Models)

#### Prototype-Guided Concept Erasure in Diffusion Models
- Link：https://arxiv.org/abs/2603.08271
- Code：

#### Pixel Motion Diffusion is What We Need for Robot Control
- Link：https://arxiv.org/abs/2509.22652
- Code：https://eronguyen.github.io/DAWN

#### CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance
- Link：https://arxiv.org/abs/2603.03281
- Code：https://hanyang-21.github.io/CFG-Ctrl

#### TAUE: Training-free Noise Transplant and Cultivation Diffusion Model
- Link：https://arxiv.org/abs/2511.02580
- Code：https://iyatomilab.github.io/TAUE

#### 3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion
- Link：https://arxiv.org/abs/2511.19117
- Code：https://github.com/work-submit/3MTI

#### DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization
- Link：https://arxiv.org/abs/2603.14267
- Code：

#### CoD: A Diffusion Foundation Model for Image Compression
- Link：https://arxiv.org/abs/2511.18706
- Code：https://github.com/microsoft/GenCodec/tree/main/CoD

#### SpiralDiff: Spiral Diffusion with LoRA for RGB-to-RAW Conversion Across Cameras
- Link：https://arxiv.org/abs/2603.14885
- Code：https://github.com/Chuancy-TJU/SpiralDiff

#### All-in-One Slider for Attribute Manipulation in Diffusion Models
- Link：https://arxiv.org/abs/2508.19195
- Code：

#### LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration
- Link：https://arxiv.org/abs/2602.20497
- Code：

#### When LoRA Betrays: Backdooring Text-to-Image Models by Masquerading as Benign Adapters
- Link：https://arxiv.org/abs/2602.21977
- Code：

#### SODA: Sensitivity-Oriented Dynamic Acceleration for Diffusion Transformer
- Link：https://arxiv.org/abs/2603.07057
- Code：https://github.com/leaves162/SODA

#### Guiding Diffusion Models with Semantically Degraded Conditions
- Link：https://arxiv.org/abs/2603.10780
- Code：https://github.com/Ming-321/Classifier-Degradation-Guidance

#### COT-FM: Cluster-wise Optimal Transport Flow Matching
- Link：https://arxiv.org/abs/2603.13395
- Code：

#### Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning
- Link：https://arxiv.org/abs/2512.24146
- Code：

#### Making Training-Free Diffusion Segmentors Scale with the Generative Power
- Link：https://arxiv.org/abs/2603.06178
- Code：https://github.com/Darkbblue/goca

#### Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation
- Link：https://arxiv.org/abs/2511.18281
- Code：https://github.com/yaramohamadi/uni-DAD

#### Refining Few-Step Text-to-Multiview Diffusion via Reinforcement Learning
- Link：https://arxiv.org/abs/2505.20107
- Code：https://github.com/ZiyiZhang27/MVC-ZigAL

#### Face2Scene: Using Facial Degradation as an Oracle for Diffusion-Based Scene Restoration
- Link：https://arxiv.org/abs/2603.16570
- Code：

#### Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens
- Link：https://arxiv.org/abs/2603.19232
- Code：https://github.com/YuqingWang1029/CubiD

#### Adaptive Auxiliary Prompt Blending for Target-Faithful Diffusion Generation
- Link：https://arxiv.org/abs/2603.19158
- Code：

#### ADAPT: Attention Driven Adaptive Prompt Scheduling and InTerpolating Orthogonal Complements for Rare Concepts Generation
- Link：https://arxiv.org/abs/2603.19157
- Code：

#### All-in-One Slider for Attribute Manipulation in Diffusion Models
- Link：https://arxiv.org/abs/2508.19195
- Code：

#### TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models
- Link：https://arxiv.org/abs/2603.17828
- Code：





#### Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models
- Link：https://arxiv.org/abs/2507.18534
- Code：https://github.com/PerceptionComputingLab/EDA

#### TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration
- Link：https://arxiv.org/abs/2603.03792
- Code：

#### CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance
- Link：https://arxiv.org/abs/2603.03281
- Code：https://hanyang-21.github.io/CFG-Ctrl

#### ConceptPrism: Concept Disentanglement in Personalized Diffusion Models via Residual Token Optimization
- Link：https://arxiv.org/abs/2602.19575
- Code：

#### SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models
- Link：https://arxiv.org/abs/2602.18993
- Code：


<a name="Vision-Transformer"></a>
# Vision Transformer

#### Make it SING: Analyzing Semantic Invariants in Classifiers
- Link：https://arxiv.org/abs/2603.14610
- Code：

#### Revisiting Model Stitching In the Foundation Model Era
- Link：https://arxiv.org/abs/2603.12433
- Code：

#### BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers
- Link：https://arxiv.org/abs/2603.09582
- Code：https://github.com/EdwardChasel/BinaryAttention











#### MuViT: Multi-Resolution Vision Transformers for Learning Across Scales in Microscopy
- Link：https://arxiv.org/abs/2602.24222
- Code：

<a name="Panoptic-Segmentation"></a>
# 全景分割(Panoptic Segmentation)
#### Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation
- Link：https://arxiv.org/abs/2603.15475
- Code：https://github.com/zyfone/EDA-PSeg



<a name="VL"></a>
# 视觉和语言(Vision-Language)

#### VL-RouterBench: A Benchmark for Vision-Language Model Routing
- Link：https://arxiv.org/abs/2512.23562
- Code：

#### CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods
- Link：https://arxiv.org/abs/2508.18753
- Code：

#### More than the Sum: Panorama-Language Models for Adverse Omni-Scenes
- Link：https://arxiv.org/abs/2603.09573
- Code：https://github.com/InSAI-Lab/PanoVQA

#### A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods
- Link：https://arxiv.org/abs/2508.18753
- Code：

#### Probing and Bridging Geometry-Interaction Cues for Affordance Reasoning in Vision Foundation Models
- Link：https://arxiv.org/abs/2602.20501
- Code：

#### ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation
- Link：https://arxiv.org/abs/2603.05530
- Code：

#### AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network
- Link：https://arxiv.org/abs/2603.12659
- Code：

#### HoneyBee: Data Recipes for Vision-Language Reasoners
- Link：https://arxiv.org/abs/2510.12225
- Code：https://huggingface.co/datasets/facebook/HoneyBee

#### HATS: Hardness-Aware Trajectory Synthesis for GUI Agents
- Link：https://arxiv.org/abs/2603.12138
- Code：



#### StructXLIP: Enhancing Vision-language Models with Multimodal Structural Cues
- Link：https://arxiv.org/abs/2602.20089
- Code：https://github.com/intelligolabs/StructXLIP

#### StructXLIP: Enhancing Vision-language Models with Multimodal Structural Cues
- Link：https://arxiv.org/abs/2503.07853
- Code：https://github.com/intelligolabs/StructXLIP

<a name="Object-Detection"></a>
# 目标检测(Object Detection)


#### Prompt-Free Universal Region Proposal Network
- Link：https://arxiv.org/abs/2603.17554
- Code：https://github.com/tangqh03/PF-RPN

#### Does YOLO Really Need to See Every Training Image in Every Epoch?
- Link：https://arxiv.org/abs/2603.17684
- Code：

#### Fourier Angle Alignment for Oriented Object Detection in Remote Sensing
- Link：https://arxiv.org/abs/2602.23790
- Code：https://github.com/gcy0423/Fourier-Angle-Alignment

#### Fourier Angle Alignment for Oriented Object Detection in Remote Sensing
- Link：https://arxiv.org/abs/2602.23790
- Code：https://github.com/gcy0423/Fourier-Angle-Alignment

#### Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection
- Link：https://arxiv.org/abs/2512.17514
- Code：

<a name="DA"></a>
## 数据增强(Data Augmentation)

#### Fixed Anchors Are Not Enough: Dynamic Retrieval and Persistent Homology for Dataset Distillation
- Link：https://arxiv.org/abs/2602.24144
- Code：


<a name="Anomaly-Detection"></a>
# 异常检测(Anomaly Detection)
#### MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection
- Link：https://arxiv.org/abs/2603.03101
- Code：https://github.com/CoCoRessa/MoECLIP

#### EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection
- Link：https://arxiv.org/abs/2603.11521
- Code：

#### RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation
- Link：https://arxiv.org/abs/2603.11106
- Code：

#### Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs
- Link：https://arxiv.org/abs/2603.02618
- Code：

#### No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection
- Link：https://arxiv.org/abs/2602.19248
- Code：https://github.com/VitaminCreed/LAVIDA

#### UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression
- Link：https://arxiv.org/abs/2509.25934
- Code：https://github.com/yuanzhao-CVLAB/UniMMAD

#### Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods
- Link：https://arxiv.org/abs/2603.15026
- Code：https://omerbenhayun.github.io/stall-video

#### VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer
- Link：https://arxiv.org/abs/2603.07952
- Code：https://github.com/7HHHHH/VisualAD

#### Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning
- Link：https://arxiv.org/abs/2603.00550
- Code：

#### Diversity over Uniformity: Rethinking Representation in Generated Image Detection
- Link：https://arxiv.org/abs/2603.00717
- Code：https://github.com/Yanmou-Hui/DoU

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
- Link：https://arxiv.org/pdf/2602.20412
- Code：

<a name="VT"></a>
# 目标跟踪(Object Tracking)
#### UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking
- Link：https://arxiv.org/abs/2602.23734
- Code：https://github.com/EIT-NLP/UTPTrack

#### Rethinking Two-Stage Referring-by-Tracking in Referring Multi-Object Tracking
- Link：https://arxiv.org/abs/2503.07516
- Code：https://github.com/buptLwz/FlexHook

#### Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking
- Link：https://arxiv.org/abs/2603.06034
- Code：

#### Changes in Real Time: Online Scene Change Detection with Multi-View Fusion
- Link：https://arxiv.org/abs/2511.12370
- Code：https://chumsy0725.github.io/O-SCD/

#### Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction
- Link：https://arxiv.org/abs/2602.18996
- Code：https://github.com/shannany0606/CCMP


<a name="Semantic-Segmentation"></a>
# 语义分割(Semantic Segmentation)

#### ACPV-Net: All-Class Polygonal Vectorization for Seamless Vector Map Generation from Aerial Imagery
- Link：https://arxiv.org/abs/2603.16671
- Code：https://github.com/HeinzJiao/ACPV-Net

#### Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels
- Link：https://arxiv.org/abs/2603.18671
- Code：https://jmlipman.github.io/SCNP-SameClassNeighborPenalization

#### MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention
- Link：https://arxiv.org/abs/2603.01361
- Code：https://github.com/spiderforest/MixerCSeg

#### Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild
- Link：https://arxiv.org/abs/2603.11618
- Code：

#### Discriminative Perception via Anchored Description for Reasoning Segmentation
- Link：https://arxiv.org/abs/2603.04002
- Code：https://github.com/mrazhou/DPAD

<a name="Instance-Segmentation"></a>
# 实例分割(Instance Segmentation)


<a name="FewShot"></a>
# 少样本学习(Few-Shot Learning)

#### Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection
- Link：https://arxiv.org/abs/2603.18541
- Code：

#### MAGIC: Few-Shot Mask-Guided Anomaly Inpainting with Prompt Perturbation, Spatially Adaptive Guidance, and Context Awareness
- Link：https://arxiv.org/abs/2507.02314
- Code：https://github.com/SpatialAILab/MAGIC-Anomaly-generation

#### SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation
- Link：https://arxiv.org/abs/2603.06572
- Code：https://github.com/Surrey-UP-Lab/SCOPE

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

#### Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in Whole-Slide Image Prognosis
- Link：https://arxiv.org/abs/2603.10526
- Code：https://github.com/liupei101/STEPH

#### CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis
- Link：https://arxiv.org/abs/2602.21637
- Code：

#### CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis
- Link：https://arxiv.org/abs/2602.21637
- Code：

#### Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code
- Link：https://arxiv.org/abs/2501.18328
- Code：https://github.com/ycwu1997/CodeBrain

#### LUMINA: A Multi-Vendor Mammography Benchmark with Energy Harmonization Protocol
- Link：https://arxiv.org/abs/2603.14644
- Code：

#### Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data
- Link：https://arxiv.org/abs/2508.01450
- Code：https://github.com/mihara-bot/DIQ

#### Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning
- Link：https://arxiv.org/abs/2603.13682
- Code：

#### X-WIN: Building Chest Radiograph World Model via Predictive Sensing
- Link：https://arxiv.org/abs/2511.14918
- Code：

#### Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning
- Link：https://arxiv.org/abs/2602.18867
- Code：

#### LUMINA: A Multi-Vendor Mammography Benchmark with Energy Harmonization Protocol
- Link：https://arxiv.org/abs/2603.14644
- Code：

#### Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images
- Link：https://arxiv.org/abs/2603.18461
- Code：https://github.com/naivete5656/CPNN

#### Benchmarking Endoscopic Surgical Image Restoration and Beyond
- Link：https://arxiv.org/abs/2505.19161
- Code：



#### MRI Contrast Enhancement Kinetics World Model
- Link：https://arxiv.org/abs/2602.19285
- Code：https://github.com/DD0922/MRI-Contrast-Enhancement-Kinetics-World-Model

#### Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning
- Link：https://arxiv.org/abs/2602.18867
- Code：

#### Solving a Nonlinear Blind Inverse Problem for Tagged MRI with Physics and Deep Generative Priors
- Link：https://arxiv.org/abs/2603.00882
- Code：

#### Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning
- Link：https://arxiv.org/abs/2603.00667
- Code：


<a name="MIS"></a>
# 医学图像分割(Medical Image Segmentation)

#### Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model
- Link：https://arxiv.org/abs/2603.05012
- Code：https://github.com/derekshiii/Tell2Adapt

#### SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering for Medical Image Segmentation
- Link：https://arxiv.org/abs/2603.11492
- Code：https://github.com/Jwei-Z/SPEGC-for-MIS

<a name="VOS"></a>
# 视频目标分割(Video Object Segmentation)
#### MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator
- Link：https://arxiv.org/abs/2512.11782
- Code：https://pq-yang.github.io/projects/MatAnyone2/

#### Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos
- Link：https://arxiv.org/abs/2503.22174
- Code：

<a name="Action-Detection"></a>
# 行为检测(Action Detection)


#### SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion
- Link：https://arxiv.org/abs/2603.12764
- Code：https://github.com/jack1ee/SAVAX

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

#### Learning Coordinate-based Convolutional Kernels for Continuous SE(3) Equivariant and Efficient Point Cloud Analysis
- Link：https://arxiv.org/abs/2603.17538
- Code：

#### Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors
- Link：https://arxiv.org/abs/2603.18782
- Code：

#### Universal 3D Shape Matching via Coarse-to-Fine Language Guidance
- Link：https://arxiv.org/abs/2511.23055
- Code：

#### QD-PCQA: Quality-Aware Domain Adaptation for Point Cloud Quality Assessment
- Link：https://arxiv.org/abs/2603.03726
- Code：

#### QD-PCQA: Quality-Aware Domain Adaptation for Point Cloud Quality Assessment
- Link：https://arxiv.org/abs/2603.03726
- Code：https://github.com/huhu-code/QD-PCQA

<a name="SSL"></a>
# 自监督学习(Self-supervised Learning)

#### BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird's-Eye View Images
- Link：https://arxiv.org/abs/2603.17159
- Code：

#### Towards Stable Self-Supervised Object Representations in Unconstrained Egocentric Video
- Link：https://arxiv.org/abs/2603.13912
- Code：

#### Bootstrap Dynamic-Aware 3D Visual Representation for Scalable Robot Learning
- Link：https://arxiv.org/abs/2512.00074
- Code：https://kolakivy.github.io/AFRO/

<a name="bio"></a>
# 生物工程(bioengineering)

#### Multimodal Protein Language Models for Enzyme Kinetic Parameters: From Substrate Recognition to Conformational Adaptation
- Link：https://arxiv.org/abs/2603.12845
- Code：


<a name="FL"></a>
# 联邦学习(Federated Learning)
#### Federated Active Learning Under Extreme Non-IID and Global Class Imbalance
- Link：https://arxiv.org/abs/2603.10341
- Code：https://github.com/chenchenzong/FairFAL

#### Domain-Skewed Federated Learning with Feature Decoupling and Calibration
- Link：https://arxiv.org/abs/2603.14238
- Code：https://github.com/mala-lab/F2DC

#### Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift
- Link：https://arxiv.org/abs/2603.01040
- Code：

#### HiLoRA: Hierarchical Low-Rank Adaptation for Personalized Federated Learning
- Link：https://arxiv.org/abs/2603.02785
- Code：

#### FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning
- Link：[https://arxiv.org/abs/2602.21399](https://arxiv.org/abs/2602.21399)
- Code：[https://github.com/alinadevkota/FedVG](https://github.com/alinadevkota/FedVG)

#### FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation
- Link：https://arxiv.org/abs/2603.04890
- Code：

<a name="IL"></a>
# 增量学习(Incremental Learning)



<a name="#3DOD"></a>
# 3D目标检测(3D Object Detection)






#### Look Before You Fuse: 2D-Guided Cross-Modal Alignment for Robust 3D Detection
- Link：https://arxiv.org/abs/2507.16861
- Code：

#### VirPro: Visual-referred Probabilistic Prompt Learning for Weakly-Supervised Monocular 3D Detection
- Link：https://arxiv.org/abs/2603.17470
- Code：

#### CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection
- Link：https://arxiv.org/abs/2603.05042
- Code：

#### R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection
- Link：https://arxiv.org/abs/2603.11566
- Code：

#### SPAN: Spatial-Projection Alignment for Monocular 3D Object Detection
- Link：https://arxiv.org/abs/2511.06702
- Code：

#### Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception
- Link：https://arxiv.org/abs/2602.19596
- Code：https://github.com/yihangtao/MVIG.git

#### SABER: Spatially Consistent 3D Universal Adversarial Objects for BEV Detectors
- Link：https://arxiv.org/abs/2505.22499
- Code：

#### VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object Detection
- Link：https://arxiv.org/abs/2603.00912
- Code：https://github.com/yangcaoai/VGGT-Det-CVPR2026

<a name="3DOD"></a>
# 3D语义分割(3D Semantic Segmentation)




<a name="Image-Editing"></a>
# 图像编辑(Image Editing)
#### Precise Object and Effect Removal with Adaptive Target-Aware Attention
- Link：https://arxiv.org/abs/2505.22636
- Code：https://zjx0101.github.io/projects/ObjectClear/

#### CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing
- Link：https://arxiv.org/abs/2603.08589
- Code：https://care-edit.github.io/

#### Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing
- Link：https://arxiv.org/abs/2603.17531
- Code：




#### Cycle-Consistent Tuning for Layered Image Decomposition
- Link：https://arxiv.org/abs/2602.20989
- Code：

#### BeautyGRPO: Aesthetic Alignment for Face Retouching via Dynamic Path Guidance and Fine-Grained Preference Modeling
- Link：https://arxiv.org/abs/2603.01163
- Code：

#### owards Source-Aware Object Swapping with Initial Noise Perturbation
- Link：https://arxiv.org/abs/2602.23697
- Code：

#### Cycle-Consistent Tuning for Layered Image Decomposition
- Link：[https://arxiv.org/abs/2602.20989](https://arxiv.org/abs/2602.20989)
- Code：[https://vcc.tech/research/2026/ImgDecom](https://vcc.tech/research/2026/ImgDecom)

#### ChordEdit: One-Step Low-Energy Transport for Image Editing
- Link：https://arxiv.org/abs/2602.19083
- Code：

#### Towards Source-Aware Object Swapping with Initial Noise Perturbation
- Link：https://arxiv.org/abs/2602.23697
- Code：

<a name="Image-Inpainting"></a>
# 图像补全/图像修复(Image Inpainting)
#### 1. HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images
- Link：https://arxiv.org/abs/2603.02210
- Code：https://correr-zhou.github.io/HiFi-Inpaint/

#### HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images
- Link：https://arxiv.org/abs/2603.02210
- Code：https://correr-zhou.github.io/HiFi-Inpaint/

<a name="GAN"></a>
# 生成对抗网络(GAN)




<a name="Video-Editing"></a>
# 视频编辑(Video Editing)

#### Object-WIPER: Training-Free Object and Associated Effect Removal in Videos
- Link：https://arxiv.org/abs/2601.06391
- Code：

#### HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles
- Link：https://arxiv.org/abs/2602.21333
- Code：https://horizonforge.github.io/

<a name="LLV"></a>
# Low-level Vision

#### ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration
- Link：https://arxiv.org/abs/2603.00906
- Code：https://github.com/Sailor-t/ShiftLUT

#### F²HDR: Two-Stage HDR Video Reconstruction via Flow Adapter and Physical Motion Modeling
- Link：https://arxiv.org/abs/2603.14920
- Code：

#### Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared
- Link：https://arxiv.org/abs/2603.08018
- Code：https://github.com/harukiv/DCMIF

#### BluRef: Unsupervised Image Deblurring with Dense-Matching References
- Link：https://arxiv.org/abs/2603.14176
- Code：https://qualcomm-ai-research.github.io/BluRef/

#### Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis
- Link：https://arxiv.org/abs/2603.12083
- Code：https://github.com/XiaolongQian/UniCAC

#### Cross-Scale Pansharpening via ScaleFormer and the PanScale Benchmark
- Link：https://arxiv.org/abs/2603.00543
- Code：

#### ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration
- Link：https://arxiv.org/abs/2603.00906
- Code：

#### Reparameterized Tensor Ring Functional Decomposition for Multi-Dimensional Data Recovery
- Link：https://arxiv.org/abs/2603.01034
- Code：https://github.com/YangyangXu2002/RepTRFD

#### Lumosaic: Hyperspectral Video via Active Illumination and Coded-Exposure Pixels
- Link：[https://arxiv.org/abs/2602.22140](https://arxiv.org/abs/2602.22140)
- Code：

#### MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision
- Link：[https://arxiv.org/abs/2602.20689](https://arxiv.org/abs/2602.20689)
- Code：[https://cvpr26-matched.github.io](https://cvpr26-matched.github.io)


#### Continuous Exposure-Time Modeling for Realistic Atmospheric Turbulence Synthesis
- Link：https://arxiv.org/abs/2603.01398
- Code：https://github.com/Jun-Wei-Zeng/ET-Turb


<a name="SR"></a>
# 超分辨率(Super-Resolution)
#### RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution
- Link：https://arxiv.org/abs/2603.12493
- Code：

#### UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution
- Link：https://arxiv.org/abs/2603.11680
- Code：

#### FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution
- Link：https://arxiv.org/abs/2603.02692
- Code：https://github.com/Ar0Kim/FiDeSR

#### Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset
- Link：https://arxiv.org/abs/2603.04745
- Code：https://github.com/JZD151/Real-IISR

#### AlignVAR: Towards Globally Consistent Visual Autoregression for Image Super-Resolution
- Link：https://arxiv.org/abs/2603.00589
- Code：

#### Spectral Super-Resolution via Adversarial Unfolding and Data-Driven Spectrum Regularization: From Multispectral Satellite Data to NASA Hyperspectral Image
- Link：https://arxiv.org/abs/2603.00920
- Code：https://sites.google.com/view/chiahsianglin/software

#### AlignVAR: Towards Globally Consistent Visual Autoregression for Image Super-Resolution
- Link：https://arxiv.org/abs/2603.00589
- Code：


<a name="Denoising"></a>
# 去噪(Denoising)





#### Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging
- Link：https://arxiv.org/abs/2603.18834
- Code：https://github.com/HeasonLee/SCGN






<a name="Image-Generation"></a>

# 图像生成(Image Generation)
#### coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation
- Link：https://arxiv.org/abs/2603.12829
- Code：

#### OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens
- Link：https://arxiv.org/abs/2603.02138
- Code：https://openvglab.github.io/OmniLottie/

#### Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards
- Link：https://arxiv.org/abs/2603.00918
- Code：https://wookiekim.github.io/SOLACE/

#### VeCoR -- Velocity Contrastive Regularization for Flow Matching
- Link：https://arxiv.org/abs/2511.18942
- Code：https://p458732.github.io/VeCoR_Project_Page/

#### Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards
- Link：https://arxiv.org/abs/2603.00918
- Code：https://wookiekim.github.io/ARC/

#### Enhancing Spatial Understanding in Image Generation via Reward Modeling
- Link：https://arxiv.org/abs/2602.24233
- Code：https://github.com/DAGroup-PKU/SpatialT2I

#### AutoDebias: Automated Framework for Debiasing Text-to-Image Models
- Link：https://arxiv.org/abs/2508.00445
- Code：


<a name="Video-Generation"></a>
# 视频生成(Video Generation)



#### Anchoring and Rescaling Attention for Semantically Coherent Inbetweening
- Link：https://arxiv.org/abs/2603.17651
- Code：https://github.com/teunchoi/TGI

#### Training-free Motion Factorization for Compositional Video Generation
- Link：https://arxiv.org/abs/2603.09104
- Code：

#### Chain of Event-Centric Causal Thought for Physically Plausible Video Generation
- Link：https://arxiv.org/abs/2603.09094
- Code：

#### FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters
- Link：https://arxiv.org/abs/2603.01685
- Code：

#### EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation
- Link：https://arxiv.org/abs/2603.12267
- Code：https://silentview.github.io/EVATok/

#### Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context
- Link：[https://arxiv.org/abs/2602.21929](https://arxiv.org/abs/2602.21929)
- Code：

#### NOVA: Sparse Control, Dense Synthesis for Pair-Free Video Editing
- Link：https://arxiv.org/abs/2603.02802
- Code：

#### CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video
- Link：https://arxiv.org/abs/2603.04291
- Code：https://lg-li.github.io/project/cubecomposer

#### FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning
- Link：https://arxiv.org/abs/2603.05506
- Code：https://weijielyu.github.io/FaceCam

#### UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation
- Link：https://arxiv.org/abs/2603.01418
- Code：

#### ExpPortrait: Expressive Portrait Generation via Personalized Representation
- Link：https://arxiv.org/abs/2602.19900
- Code：

#### LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation
- Link：https://arxiv.org/abs/2510.08318
- Code：

#### FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters
- Link：https://arxiv.org/abs/2603.01685
- Code：无

#### Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models
- Link：[https://arxiv.org/abs/2602.20981](https://arxiv.org/abs/2602.20981)
- Code：

#### The devil is in the details: Enhancing Video Virtual Try-On via Keyframe-Driven Details Injection
- Link：https://arxiv.org/abs/2512.20340
- Code：

<a name="3D-Generation"></a>
# 3D生成




#### Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing
- Link：https://arxiv.org/abs/2603.17583
- Code：

#### NI-Tex: Non-isometric Image-based Garment Texture Generation
- Link：https://arxiv.org/abs/2511.18765
- Code：

#### ForgeDreamer: Industrial Text-to-3D Generation with Multi-Expert LoRA and Cross-View Hypergraph
- Link：https://arxiv.org/abs/2603.09266
- Code：

#### Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image
- Link：https://arxiv.org/abs/2603.14772
- Code：

#### BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation
- Link：https://arxiv.org/abs/2602.18873
- Code：https://wangmiaowei.github.io/BiMotion.github.io/

#### Mesh-Pro: Asynchronous Advantage-guided Ranking Preference Optimization for Artist-style Quadrilateral Mesh Generation
- Link：https://arxiv.org/abs/2603.00526
- Code：

#### MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing
- Link：https://arxiv.org/abs/2601.00204
- Code：https://xiaokunsun.github.io/MorphAny3D.github.io

#### Easy3E: Feed-Forward 3D Asset Editing via Rectified Voxel Flow
- Link：[https://arxiv.org/abs/2602.21499](https://arxiv.org/abs/2602.21499)
- Code：

#### BiMotion: B-spline Motion for Text-guided Dynamic 3D Character Generation
- Link：https://arxiv.org/abs/2602.18873
- Code：

<a name="Video-Understanding"></a>
# 视频理解(Video Understanding)

#### F2HDR: Two-Stage HDR Video Reconstruction via Flow Adapter and Physical Motion Modeling
- Link：https://arxiv.org/abs/2603.14920
- Code：

#### Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods
- Link：https://arxiv.org/abs/2603.15026
- Code：https://omerbenhayun.github.io/stall-video

#### PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning
- Link：https://arxiv.org/abs/2602.20537
- Code：https://github.com/fhjdqaq/PFGNet

#### Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding
- Link：https://arxiv.org/abs/2603.00512
- Code：https://github.com/MAC-AutoML/WFS-SB

#### Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding
- Link：https://arxiv.org/abs/2603.15167
- Code：

#### StreamingTOM: Streaming Token Compression for Efficient Video Understanding
- Link：https://arxiv.org/abs/2510.18269
- Code：https://yige24.github.io/StreamingTOM

#### Follow the Saliency: Supervised Saliency for Retrieval-augmented Dense Video Captioning
- Link：https://arxiv.org/abs/2603.11460
- Code：https://github.com/ermitaju1/STaRC

#### Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning
- Link：https://arxiv.org/abs/2603.11439
- Code：

#### VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding
- Link：https://arxiv.org/abs/2603.07071
- Code：

#### StreamReady: Learning What to Answer and When in Long Streaming Videos
- Link：https://arxiv.org/abs/2603.08620
- Code：

#### SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning
- Link：https://arxiv.org/abs/2603.05437
- Code：









#### LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding
- Link：[https://arxiv.org/abs/2602.20913](https://arxiv.org/abs/2602.20913)
- Code：[https://github.com/qiujihao19/LongVideo-R1](https://github.com/qiujihao19/LongVideo-R1)

#### VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning
- Link：https://arxiv.org/abs/2511.19524
- Code：

#### SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning
- Link：https://arxiv.org/abs/2603.05437
- Code：

#### Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding
- Link：https://arxiv.org/abs/2603.04977
- Code：https://github.com/Haorane/VideoHV-Agent

#### ViterbiPlanNet: Injecting Procedural Knowledge via Differentiable Viterbi for Planning in Instructional Videos
- Link：https://arxiv.org/abs/2603.04265
- Code：

#### Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding
- Link：https://arxiv.org/abs/2603.00512
- Code：

#### Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction: Dataset, Model and Benchmark
- Link：https://arxiv.org/abs/2603.00611
- Code：https://github.com/nju-cite/DynaSpec

#### Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding
- Link：https://arxiv.org/abs/2602.18977
- Code：https://github.com/th-nesh/Frame2Freq

#### FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding
- Link：https://arxiv.org/abs/2603.02096
- Code：https://yiwengxie.com/FluxMem/

#### Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models
- Link：https://arxiv.org/abs/2603.01400
- Code：https://tyroneli.github.io/AOT



<a name="3D-Human-Pose-Estimation"></a>
# 3D人体姿态估计(3D Human Pose Estimation)
#### GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion
- Link：https://arxiv.org/abs/2603.17161
- Code：https://caizhuojiang.github.io/GazeOnce360/

#### OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery
- Link：https://arxiv.org/abs/2603.17355
- Code：https://tsukasane.github.io/Video-OnlineHMR/

#### Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation
- Link：https://arxiv.org/abs/2511.22184
- Code：https://github.com/dqj5182/FECO_RELEASE

#### Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation
- Link：https://arxiv.org/abs/2501.05264
- Code：https://github.com/MICLAB-BUPT/AWC

#### Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator
- Link：https://arxiv.org/abs/2603.14726
- Code：

#### CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation
- Link：https://arxiv.org/abs/2603.09418
- Code：https://github.com/53mins/CIGPose

#### EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR
- Link：https://arxiv.org/abs/2603.04090
- Code：

#### Sketch2Colab: Sketch-Conditioned Multi-Human Animation via Controllable Flow Distillation
- Link：https://arxiv.org/abs/2603.02190
- Code：

#### SIMSPINE: A Biomechanics-Aware Simulation Framework for 3D Spine Motion Annotation and Benchmarking
- Link：[https://arxiv.org/abs/2602.20792](https://arxiv.org/abs/2602.20792)
- Code：

#### VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery
- Link：https://arxiv.org/abs/2602.19180
- Code：

<a name="CL"></a>
# 持续学习(Continual Learning)

#### Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment
- Link：https://arxiv.org/abs/2603.10929
- Code：https://github.com/yfqi/lifelong_mlr_ifa

#### Elastic Weight Consolidation Done Right for Continual Learning
- Link：https://arxiv.org/abs/2603.18596
- Code：

<a name="Action-Recognition"></a>
# 行为识别(Action Recognition)

#### BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment
- Link：https://arxiv.org/abs/2602.19170
- Code：

#### Test-time Ego-Exo-centric Adaptation for Action Anticipation via Multi-Label Prototype Growing and Dual-Clue Consistency
- Link：https://arxiv.org/abs/2603.09798
- Code：https://github.com/ZhaofengSHI/DCPGN


<a name="KD"></a>
# 知识蒸馏(Knowledge Distillation)
#### Momentum Memory for Knowledge Distillation in Computational Pathology
- Link：[https://arxiv.org/abs/2602.21395](https://arxiv.org/abs/2602.21395)
- Code：

#### WaDi: Weight Direction-aware Distillation for One-step Image Synthesis
- Link：https://arxiv.org/abs/2603.08258
- Code：https://github.com/gudaochangsheng/WaDi

#### Fixed Anchors Are Not Enough: Dynamic Retrieval and Persistent Homology for Dataset Distillation
- Link：https://arxiv.org/abs/2602.24144
- Code：

#### Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation
- Link：[https://arxiv.org/abs/2602.19863](https://arxiv.org/abs/2602.19863)
- Code：[https://wolfilip.github.io/DEO/](https://wolfilip.github.io/DEO/)

#### Distilling Balanced Knowledge from a Biased Teacher
- Link：https://arxiv.org/abs/2506.18496
- Code：

#### Momentum Memory for Knowledge Distillation in Computational Pathology
- Link：https://arxiv.org/abs/2602.21395
- Code：https://github.com/CAIR-LAB-WFUSM/MoMKD


<a name="IC"></a>
# 图像压缩(Image Compression)

#### SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation
- Link：https://arxiv.org/abs/2603.07789
- Code：https://github.com/zx-pan/SGI

#### Parallax to Align Them All: An OmniParallax Attention Mechanism for Distributed Multi-View Image Compression
- Link：https://arxiv.org/abs/2603.03615
- Code：

<a name="ZSL"></a>
# Zero-Shot Learning(零样本学习)
#### Learning through Creation: A Hash-Free Framework for On-the-Fly Category Discovery
- Link：https://arxiv.org/abs/2603.13858
- Code：https://github.com/brandinzhang/LTC

#### TALON: Test-time Adaptive Learning for On-the-Fly Category Discovery
- Link：https://arxiv.org/abs/2603.08075
- Code：https://github.com/ynanwu/TALON

#### Learning through Creation: A Hash-Free Framework for On-the-Fly Category Discovery
- Link：https://arxiv.org/abs/2603.13858
- Code：https://github.com/brandinzhang/LTC

#### Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting
- Link：https://arxiv.org/abs/2603.16129
- Code：


<a name="Stereo-Matching"></a>
# 立体匹配(Stereo Matching)
#### PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts
- Link：https://arxiv.org/abs/2603.01650
- Code：

#### PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts
- Link：https://arxiv.org/abs/2603.01650
- Code：

#### Pip-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching
- Link：https://arxiv.org/abs/2602.19112
- Code：https://github.com/XPENG-Aridge-AI


<a name="SGG"></a>
# 场景图生成(Scene Graph Generation)

#### DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime
- Link：https://arxiv.org/abs/2603.10538
- Code：

<a name="Counting"></a>
# 计数(Counting)

#### UNICBench: UNIfied Counting Benchmark for MLLM
- Link：https://arxiv.org/abs/2603.00595
- Code：

<a name="INR"></a>
# 隐式神经表示(Implicit Neural Representations)




<a name="IQA"></a>
# 图像质量评价(Image Quality Assessment)

#### How to Take a Memorable Picture? Empowering Users with Actionable Feedback
- Link：[https://arxiv.org/abs/2602.21877](https://arxiv.org/abs/2602.21877)
- Code：[https://laitifranz.github.io/MemCoach/](https://laitifranz.github.io/MemCoach/)

#### Fine-grained Image Aesthetic Assessment: Learning Discriminative Scores from Relative Ranks
- Link：https://arxiv.org/abs/2603.03907
- Code：

<a name="Video-Quality-Assessment"></a>
# 视频质量评价(Video Quality Assessment)



<a name="Datasets"></a>
# 数据集(Datasets)
#### E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought
- Link：[https://arxiv.org/abs/2602.21698](https://arxiv.org/abs/2602.21698)
- Code：[https://github.com/4mm7/E-comIQ-ZH](https://github.com/4mm7/E-comIQ-ZH)

#### Continuous Exposure-Time Modeling for Realistic Atmospheric Turbulence Synthesis
- Link：https://arxiv.org/abs/2603.01398
- Code：https://github.com/Jun-Wei-Zeng/ET-Turb

#### LenghuSky-8: An 8-Year All-Sky Cloud Dataset with Star-Aware Masks and Alt-Az Calibration for Segmentation and Nowcasting
- Link：https://arxiv.org/abs/2603.16429
- Code：

#### AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models
- Link：https://arxiv.org/abs/2506.09082
- Code：





<a name="Unlearning"></a>
# 反学习(Machine Unlearning)
#### SineProject: Machine Unlearning for Stable Vision Language Alignment
- Link：https://arxiv.org/abs/2511.18444
- Code：

#### RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models
- Link：https://arxiv.org/abs/2603.14819
- Code：

#### Stake the Points: Structure-Faithful Instance Unlearning
- Link：https://arxiv.org/abs/2603.12915
- Code：





<a name="New-Tasks"></a>
# 新任务(New Tasks)



<a name="Improving-Reasoning"></a>
# 模型加速(Improving Reasoning)
#### Flash-Unified: A Training-Free and Task-Aware Acceleration Framework for Native Unified Models
- Link：https://arxiv.org/abs/2603.15271
- Code：https://github.com/Rirayh/FlashU


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
#### Stable Spike: Dual Consistency Optimization via Bitwise AND Operations for Spiking Neural Networks
- Link：https://arxiv.org/abs/2603.11676
- Code：

#### Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model
- Link：https://arxiv.org/abs/2410.07547
- Code：https://github.com/hzc1208/HD_LIF

<a name="IRetrieval"></a>
# 图像检索

#### TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking
- Link：https://arxiv.org/abs/2602.18863
- Code：

#### PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing
- Link：https://arxiv.org/abs/2603.04598
- Code：

#### NaiLIA: Multimodal Nail Design Retrieval Based on Dense Intent Descriptions and Palette Queries
- Link：https://arxiv.org/abs/2603.05446
- Code：

# 其他(Others)
#### NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training
- Link：https://arxiv.org/abs/2602.22059
- Code：

#### Defending Unauthorized Model Merging via Dual-Stage Weight Protection
- Link：https://arxiv.org/abs/2511.11851
- Code：

#### BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning
- Link：https://arxiv.org/abs/2603.03920
- Code：

#### ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation
- Link：https://arxiv.org/abs/2603.02945
- Code：

#### GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization through EM-Guided Decomposition and Temporal Refinement
- Link：https://arxiv.org/abs/2603.05095
- Code：

#### DC-Merge: Improving Model Merging with Directional Consistency
- Link：https://arxiv.org/abs/2603.06242
- Code：https://github.com/Tobeginwith/DC-Merge

#### Defending Unauthorized Model Merging via Dual-Stage Weight Protection
- Link：https://arxiv.org/abs/2511.11851
- Code：

#### BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning
- Link：https://arxiv.org/abs/2603.03920
- Code：

#### Bridging Domains through Subspace-Aware Model Merging
- Link：https://arxiv.org/abs/2603.05768
- Code：

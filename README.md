<div align="center">

# 🔐 Privacy-Aware Embodied AI: A Position Paper

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![arXiv](https://img.shields.io/badge/arXiv-2510.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2510.xxxxx)
[![GitHub stars](https://img.shields.io/github/stars/JL-F-D/Survey?style=social)](https://github.com/JL-F-D/Survey)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/JL-F-D/Survey/graphs/commit-activity)

</div>

<div align="center">
  <img src="assets/banner.png" width="100%" alt="Survey Banner"/>
</div>

---

<p align="center">
  <b>🔥 This is a curated paper list of "Privacy-Aware Embodied AI: A Position Paper".</b>
  <br>
  To the best of our knowledge, this work presents the <b>first comprehensive survey</b> on privacy considerations in Embodied AI systems, covering the entire closed-loop framework from instruction understanding to environment perception to action planning.
  <br>
  We will continue to <b>UPDATE</b> this repository with the latest developments! 
  <br>
  ⭐ <b>Star us to stay tuned!</b> 😘
</p>

---

## 📢 News

| Date | News |
|:-----|:-----|
| 🔥 2025/06 | Our position paper is released on arXiv! |
| 🔥 2025/06 | Repository created! |

---

## 🔍 Table of Contents

- [📖 Introduction](#-introduction)
- [🏗️ Taxonomy](#️-taxonomy)
- [💬 Instruction Understanding](#-instruction-understanding)
  - [Privacy-Preserving Input](#privacy-preserving-input)
  - [Instruction-Side Safety](#instruction-side-safety)
  - [Instruction Grounding Accuracy](#instruction-grounding-accuracy)
  - [Instruction Fulfillment Utility](#instruction-fulfillment-utility)
  - [Instruction Parsing Efficiency](#instruction-parsing-efficiency)
  - [Robust Instruction Understanding](#robust-instruction-understanding)
  - [Instruction Generalization](#instruction-generalization)
  - [Human-Centered Instruction Understanding](#human-centered-instruction-understanding)
- [👁️ Environment Perception](#️-environment-perception)
  - [Privacy-Preserving Perception](#privacy-preserving-perception)
  - [Adversarial Perception Defense](#adversarial-perception-defense)
  - [Perception Accuracy](#perception-accuracy)
  - [Perception Utility](#perception-utility)
  - [Perception Efficiency](#perception-efficiency)
  - [Robust Perception](#robust-perception)
  - [Generalizable Perception](#generalizable-perception)
  - [Human-Aware Perception](#human-aware-perception)
- [🎯 Action Planning](#-action-planning)
  - [Safety Constraints](#safety-constraints)
  - [Privacy-Aware Planning](#privacy-aware-planning)
  - [Robust Planning](#robust-planning)
- [🔮 Challenges and Future Directions](#-challenges-and-future-directions)
- [🔖 Citation](#-citation)
- [📧 Contact](#-contact)

---

## 📖 Introduction

<div align="center">
  <img src="assets/overview.png" width="90%" alt="Survey Overview"/>
  <br>
  <em>Fig. 1: Overview of Privacy-Aware Embodied AI.</em>
</div>

<br>

Embodied artificial intelligence (EAI) represents a critical frontier in bridging large-scale models with physical-world interaction. While existing EAI models demonstrate remarkable capabilities, their real-world deployment faces severe constraints due to fundamental privacy concerns. This position paper addresses this gap through the **first comprehensive review of privacy-aware EAI** across the entire data-model-training lifecycle.

---

## 🏗️ Taxonomy

<div align="center">
  <img src="assets/taxonomy.png" width="95%" alt="Taxonomy"/>
  <br>
  <em>Fig. 2: Privacy-aware embodied AI evaluation framework spanning four phases: Instruction Understanding, Environment Perception, Action Planning, and Physical Interaction.</em>
</div>
### Overview Table

| Phase | Dimension | Key Focus | Representative Works |
|:------|:----------|:----------|:---------------------|
| **I. Instruction Understanding** | Privacy-Preserving Input | Federated learning, local DP for NLU | FedVLN, FedVLA, Privacy-BERT |
| | Instruction-Side Safety | Jailbreak defense, safety filtering | AGENTSAFE, BadNAVer, POEX |
| | Grounding Accuracy | Instruction-trajectory alignment | Waypoint Planner, DoRO |
| | Fulfillment Utility | Ambiguity resolution, preference modeling | JARVIS, REI-Bench |
| | Parsing Efficiency | Sub-instruction decomposition, distillation | MoLe-VLA, MAGIC |
| | Robust Understanding | Noise/adversarial tolerance | Pragmatic ToM, NavA³ |
| | Generalization | Zero-shot, open-vocabulary grounding | RT-2, ZSON, OpenMap |
| | Human-Centered | Dialog, clarification, user modeling | DialFRED, Mixed-Initiative Dialog |
| **II. Environment Perception** | Privacy-Preserving Perception | Low-resolution, anonymization, FL | Ultra-Low-Res RGB, FLAME, FedVLN |
| | Adversarial Defense | Attack detection, active defense | VPR Attacks, BadDepth, Embodied Active Defense |
| | Perception Accuracy | Multi-modal fusion, calibration | TVT-Transformer, EmbodiedScan |
| | Perception Utility | Task-driven encoding, token pruning | VLA-Pruner, CompressorVLA |
| | Perception Efficiency | Event cameras, semantic maps | HALSIE, MapNav, CODEI |
| | Robust Perception | Noise/occlusion tolerance | RobustNav, CronusVLA |
| | Generalizable Perception | Domain adaptation, visual reversion | ReVLA, RT-2, EnvDrop |
| | Human-Aware Perception | Social navigation, dynamic humans | HA-VLN 2.0, VLM-Social-Nav |
| **III. Action Planning** | Safety Constraints | Constrained RL, safety shielding | SafeVLA, Safety-Aware Task Planning |
| | Privacy-Aware Planning | Trajectory privacy, DP in MARL | PANav, Robots as Double Agents |
| | Robust Planning | Deviation recovery, adversarial defense | Perturbation-Aware CL, VoxPoser |
| **IV. Physical Interaction** | Reliability | Execution monitoring, failure recovery | — |
| | Attack/Abuse | Physical adversarial attacks | — |
| | Privacy | Sensor data leakage in deployment | — |
| | Value Alignment | Human preference alignment | GRAPE |

---
---

## 💬 Instruction Understanding

### Privacy-Preserving Input

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | arXiv | [FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts](https://arxiv.org/abs/2508.02190) | - |
| 2025 | arXiv | [ReVision: A Dataset and Baseline VLM for Privacy-Preserving Task-Oriented Visual Instruction Rewriting](https://arxiv.org/abs/2502.14780) | - |
| 2024 | ICLR | [Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory](https://proceedings.iclr.cc/paper_files/paper/2024/file/08305d8b2ddab98932c163ea73df065f-Paper-Conference.pdf) | - |
| 2022 | ECCV | [FedVLN: Privacy-Preserving Federated Vision-and-Language Navigation](https://arxiv.org/abs/2203.14936) | - |
| 2021 | CIKM | [Natural Language Understanding with Privacy-Preserving BERT](https://doi.org/10.1145/3459637.3482281) | - |

### Instruction-Side Safety

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | arXiv | [AGENTSAFE: Benchmarking the Safety of Embodied Agents on Hazardous Instructions](https://arxiv.org/abs/2506.14697) | - |
| 2025 | arXiv | [Advancing Embodied Agent Security: From Safety Benchmarks to Input Moderation](https://arxiv.org/abs/2504.15699) | - |
| 2025 | arXiv | [BadNAVer: Exploring Jailbreak Attacks On Vision-and-Language Navigation](https://arxiv.org/abs/2505.12443) | - |
| 2025 | arXiv | [POEX: Towards Policy Executable Jailbreak Attacks Against the LLM-based Robots](https://arxiv.org/abs/2412.16633) | - |

### Instruction Grounding Accuracy

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2024 | PAA | [Instruction-aligned hierarchical waypoint planner for vision-and-language navigation](https://api.semanticscholar.org/CorpusID:273112653) | - |
| 2024 | IROS | [Mind the Error! Detection and Localization of Instruction Errors in Vision-and-Language Navigation](https://doi.org/10.1109/IROS58592.2024.10801822) | - |
| 2024 | RA-L | [Learning Multimodal Confidence for Intention Recognition in Human-Robot Interaction](https://doi.org/10.1109/LRA.2024.3432352) | - |
| 2023 | ROBIO | [A Modular Framework for Robot Embodied Instruction Following by Large Language Model](https://doi.org/10.1109/ROBIO58561.2023.10355013) | - |
| 2021 | arXiv | [Learning a natural-language to LTL executable semantic parser for grounded robotics](https://arxiv.org/abs/2008.03277) | - |
| 2019 | ACL | [Stay on the Path: Instruction Fidelity in Vision-and-Language Navigation](https://aclanthology.org/P19-1181/) | - |
| 2018 | Autonomous Robots | [Grounding natural language instructions to semantic goal representations](https://api.semanticscholar.org/CorpusID:41441732) | - |

### Instruction Fulfillment Utility

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | arXiv | [JARVIS: A Neuro-Symbolic Commonsense Reasoning Framework for Conversational Embodied Agents](https://arxiv.org/abs/2208.13266) | - |
| 2025 | arXiv | [REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?](https://arxiv.org/abs/2505.10872) | - |
| 2025 | arXiv | [A Model-Agnostic Approach for Semantically Driven Disambiguation in Human-Robot Interaction](https://arxiv.org/abs/2409.17004) | - |
| 2023 | arXiv | [Open-Ended Instructable Embodied Agents with Memory-Augmented Large Language Models](https://arxiv.org/abs/2310.15127) | - |
| 2022 | RA-L | [DoRO: Disambiguation of Referred Object for Embodied Agents](https://doi.org/10.1109/LRA.2022.3195198) | - |

### Instruction Parsing Efficiency

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | arXiv | [MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers](https://arxiv.org/abs/2503.20384) | - |
| 2025 | Applied Intelligence | [A multilevel attention network with sub-instructions for continuous vision-and-language navigation](http://dx.doi.org/10.1007/s10489-025-06544-9) | - |
| 2025 | RA-L | [Boosting Efficient Reinforcement Learning for Vision-and-Language Navigation With Open-Sourced LLM](https://doi.org/10.1109/LRA.2024.3511402) | - |
| 2024 | arXiv | [MAGIC: Meta-Ability Guided Interactive Chain-of-Distillation for Effective-and-Efficient VLN](https://arxiv.org/abs/2406.17960) | - |
| 2023 | arXiv | [Grounding Language with Visual Affordances over Unstructured Data](https://arxiv.org/abs/2210.01911) | - |
| 2022 | RA-L | [COSM2IC: Optimizing Real-Time Multi-Modal Instruction Comprehension](https://doi.org/10.1109/LRA.2022.3194683) | - |

### Robust Instruction Understanding

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | arXiv | [NavA³: Understanding Any Instruction, Navigating Anywhere, Finding Anything](https://arxiv.org/abs/2508.04598) | - |
| 2025 | arXiv | [Pragmatic Embodied Spoken Instruction Following in Human-Robot Collaboration with Theory of Mind](https://arxiv.org/abs/2409.10849) | - |
| 2025 | arXiv | [Learning Efficient and Robust Language-conditioned Manipulation using Textual-Visual Relevancy](https://arxiv.org/abs/2406.15677) | - |
| 2025 | CoRL Workshop | [Task Robustness via Re-Labelling Vision-Action Robot Data](https://openreview.net/forum?id=M6M5W0lmaY) | - |
| 2024 | arXiv | [Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models](https://arxiv.org/abs/2405.19802) | - |
| 2020 | IJRR | [Multimodal estimation and communication of latent semantic knowledge for robust execution](https://doi.org/10.1177/0278364920917755) | - |

### Instruction Generalization

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | ACM MM | [OpenMap: Instruction Grounding via Open-Vocabulary Visual-Language Mapping](http://dx.doi.org/10.1145/3746027.3754887) | - |
| 2023 | arXiv | [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818) | - |
| 2023 | arXiv | [Instruction-Following Agents with Multimodal Transformer](https://arxiv.org/abs/2210.13431) | - |
| 2023 | arXiv | [ZSON: Zero-Shot Object-Goal Navigation using Multimodal Goal Embeddings](https://arxiv.org/abs/2206.12403) | - |

### Human-Centered Instruction Understanding

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | Book Chapter | [Human-Centred Robotics and AI for Trustworthy Human-Robot Interaction](https://doi.org/10.1007/978-3-031-97673-5_9) | - |
| 2025 | arXiv | [Mixed-Initiative Dialog for Human-Robot Collaborative Manipulation](https://arxiv.org/abs/2508.05535) | - |
| 2025 | arXiv | [Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning](https://arxiv.org/abs/2504.00907) | - |
| 2024 | HRI | [Understanding Large-Language Model (LLM)-powered Human-Robot Interaction](http://dx.doi.org/10.1145/3610977.3634966) | - |
| 2023 | arXiv | [Human-Centric Autonomous Systems With LLMs for User Command Reasoning](https://arxiv.org/abs/2311.08206) | - |
| 2023 | SIGIR | [Evaluating Task-oriented Dialogue Systems with Users](https://doi.org/10.1145/3539618.3591788) | - |
| 2022 | RA-L | [DialFRED: Dialogue-Enabled Agents for Embodied Instruction Following](http://dx.doi.org/10.1109/LRA.2022.3193254) | - |
| 2022 | arXiv | [Dialog Acts for Task-Driven Embodied Agents](https://arxiv.org/abs/2209.12953) | - |
| 2018 | RoboDIAL | [Jointly Improving Parsing and Perception for Natural Language Commands through Human-Robot Dialog](http://nn.cs.utexas.edu/pub-view.php?PubID=127720) | - |

---

## 👁️ Environment Perception

### Privacy-Preserving Perception

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | arXiv | [Improved Semantic Segmentation from Ultra-Low-Resolution RGB Images Applied to Privacy-Preserving Object-Goal Navigation](https://arxiv.org/abs/2507.16034) | - |
| 2025 | arXiv | [Real-Time Privacy Preservation for Robot Visual Perception](https://arxiv.org/abs/2505.05519) | - |
| 2025 | arXiv | [Privacy Risks of Robot Vision: A User Study on Image Modalities and Resolution](https://arxiv.org/abs/2505.07766) | - |
| 2025 | Applied Sciences | [Privacy-Preserved Visual Simultaneous Localization and Mapping Based on a Dual-Component Approach](https://www.mdpi.com/2076-3417/15/5/2583) | - |
| 2024 | J. Responsible Technology | [Inherently privacy-preserving vision for trustworthy autonomous systems: Needs and solutions](https://doi.org/10.1016/j.jrt.2024.100079) | - |
| 2024 | arXiv | [DT-RaDaR: Digital Twin Assisted Robot Navigation using Differential Ray-Tracing](https://arxiv.org/abs/2411.12284) | - |
| 2019 | IROS | [Privacy-Preserving Robot Vision with Anonymized Faces by Extreme Low Resolution](https://doi.org/10.1109/IROS40897.2019.8967681) | - |

### Adversarial Perception Defense

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | arXiv | [Adversarial Attacks and Detection in Visual Place Recognition for Safer Robot Navigation](https://arxiv.org/abs/2506.15988) | - |
| 2025 | arXiv | [BadDepth: Backdoor Attacks Against Monocular Depth Estimation in the Physical World](https://arxiv.org/abs/2505.16154) | - |
| 2024 | arXiv | [Hijacking Vision-and-Language Navigation Agents with Adversarial Environmental Attacks](https://arxiv.org/abs/2412.02795) | - |
| 2024 | arXiv | [Malicious Path Manipulations via Exploitation of Representation Vulnerabilities of VLN Systems](https://arxiv.org/abs/2407.07392) | - |
| 2024 | arXiv | [Embodied Active Defense: Leveraging Recurrent Feedback to Counter Adversarial Patches](https://arxiv.org/abs/2404.00540) | - |

### Perception Accuracy

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | Information Fusion | [TVT-Transformer: A Tactile-visual-textual fusion network for object recognition](https://doi.org/10.1016/j.inffus.2025.102943) | - |
| 2025 | IEEE TCYB | [Collaborative Multimodal Fusion Network for Multiagent Perception](https://doi.org/10.1109/TCYB.2024.3491756) | - |
| 2025 | IEEE TIM | [Active SLAM With Dynamic Viewpoint Optimization for Robust Visual Navigation](https://doi.org/10.1109/TIM.2025.3579846) | - |
| 2024 | arXiv | [Embodied Uncertainty-Aware Object Segmentation](https://arxiv.org/abs/2408.04760) | - |
| 2024 | arXiv | [Enhancing Embodied Object Detection through Language-Image Pre-training and Implicit Object Memory](https://arxiv.org/abs/2402.03721) | - |
| 2022 | RA-L | [TransCG: A Large-Scale Real-World Dataset for Transparent Object Depth Completion and a Grasping Baseline](http://dx.doi.org/10.1109/LRA.2022.3183256) | - |
| 2021 | ICME | [Depth-Guided AdaIN and Shift Attention Network for Vision-And-Language Navigation](https://doi.org/10.1109/ICME51207.2021.9428422) | - |
| 2021 | arXiv | [Neighbor-view Enhanced Model for Vision and Language Navigation](https://arxiv.org/abs/2107.07201) | - |

### Perception Utility

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | ACM TOMM | [RA-MOSAIC: Resource Adaptive Edge AI Optimization over Spatially Multiplexed Video Streams](https://doi.org/10.1145/3715133) | - |
| 2024 | RTSS | [FLEX: Adaptive Task Batch Scheduling with Elastic Fusion in Multi-Modal Multi-View Machine Perception](https://doi.org/10.1109/RTSS62706.2024.00033) | - |
| 2024 | IJIRA | [Towards real-time embodied AI agent: a bionic visual encoding framework for mobile robotics](https://doi.org/10.1007/s41315-024-00363-w) | - |
| 2022 | IJCV | [Active Perception for Visual-Language Navigation](https://doi.org/10.1007/s11263-022-01721-6) | - |

### Perception Efficiency

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | arXiv | [MapNav: A Novel Memory Representation via Annotated Semantic Maps for Vision-and-Language Navigation](https://arxiv.org/abs/2502.13451) | - |
| 2025 | IEEE TRO | [CODEI: Resource-Efficient Task-Driven Co-Design of Perception and Decision Making for Mobile Robots](http://dx.doi.org/10.1109/TRO.2025.3552347) | - |
| 2025 | IEEE TCASAI | [EvGNN: An Event-Driven Graph Neural Network Accelerator for Edge Vision](http://dx.doi.org/10.1109/TCASAI.2024.3520905) | - |
| 2024 | Nature | [Low-latency automotive vision with event cameras](https://doi.org/10.1038/s41586-024-07409-w) | - |
| 2023 | arXiv | [HALSIE: Hybrid Approach to Learning Segmentation by Simultaneously Exploiting Image and Event Modalities](https://arxiv.org/abs/2211.10754) | - |

### Robust Perception

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | ACM FSE | [VLATest: Testing and Evaluating Vision-Language-Action Models for Robotic Manipulation](http://dx.doi.org/10.1145/3729343) | - |
| 2025 | arXiv | [CronusVLA: Towards Efficient and Robust Manipulation via Multi-Frame Vision-Language-Action Modeling](https://arxiv.org/abs/2506.19816) | - |
| 2023 | Sensors | [Regularized Denoising Masked Visual Pretraining for Robust Embodied PointGoal Navigation](https://www.mdpi.com/1424-8220/23/7/3553) | - |
| 2022 | RA-L | [TransCG: A Large-Scale Real-World Dataset for Transparent Object Depth Completion and a Grasping Baseline](http://dx.doi.org/10.1109/LRA.2022.3183256) | - |
| 2021 | arXiv | [RobustNav: Towards Benchmarking Robustness in Embodied Navigation](https://arxiv.org/abs/2106.04531) | - |

### Generalizable Perception

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | arXiv | [ReVLA: Reverting Visual Domain Limitation of Robotic Foundation Models](https://arxiv.org/abs/2409.15250) | - |
| 2023 | arXiv | [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818) | - |
| 2021 | arXiv | [Vision-Language Navigation with Random Environmental Mixup](https://arxiv.org/abs/2106.07876) | - |
| 2019 | arXiv | [Learning to Navigate Unseen Environments: Back Translation with Environmental Dropout](https://arxiv.org/abs/1904.04195) | - |

### Human-Aware Perception

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | arXiv | [HA-VLN 2.0: An Open Benchmark and Leaderboard for Human-Aware Navigation](https://arxiv.org/abs/2503.14229) | - |
| 2025 | arXiv | [Vi-LAD: Vision-Language Attention Distillation for Socially-Aware Robot Navigation](https://arxiv.org/abs/2503.09820) | - |
| 2025 | arXiv | [Unifying Large Language Model and Deep Reinforcement Learning for Human-in-Loop Interactive Socially-aware Navigation](https://arxiv.org/abs/2403.15648) | - |
| 2024 | arXiv | [Human-Aware Vision-and-Language Navigation: Bridging Simulation to Reality with Dynamic Human Interactions](https://arxiv.org/abs/2406.19236) | - |
| 2024 | arXiv | [VLM-Social-Nav: Socially Aware Robot Navigation through Scoring using Vision-Language Models](https://arxiv.org/abs/2404.00210) | - |

---

## 🎯 Action Planning

### Safety Constraints

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2025 | NeurIPS | [SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning](https://arxiv.org/abs/2503.03480) | - |
| 2025 | arXiv | [Safety Aware Task Planning via Large Language Models in Robotics](https://arxiv.org/abs/2503.15707) | - |
| 2024 | RA-L | [Manipulating Neural Path Planners via Slight Perturbations](http://dx.doi.org/10.1109/LRA.2024.3387131) | - |
| 2024 | ICRA | [Characterizing Physical Adversarial Attacks on Robot Motion Planners](https://doi.org/10.1109/ICRA57147.2024.10610344) | - |
| 2021 | RSS | [Safe Reinforcement Learning via Statistical Model Predictive Shielding](https://api.semanticscholar.org/CorpusID:235651403) | - |

### Privacy-Aware Planning

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2024 | arXiv | [PANav: Toward Privacy-Aware Robot Navigation via Vision-Language Models](https://arxiv.org/abs/2410.04302) | - |
| 2023 | arXiv | [Robots as AI Double Agents: Privacy in Motion Planning](https://arxiv.org/abs/2308.03385) | - |
| 2023 | arXiv | [Differential Privacy in Cooperative Multiagent Planning](https://arxiv.org/abs/2301.08811) | - |
| 2023 | IoTDI | [adaPARL: Adaptive Privacy-Aware Reinforcement Learning for Sequential Decision Making Human-in-the-Loop Systems](http://dx.doi.org/10.1145/3576842.3582325) | - |

### Robust Planning

| Year | Venue | Paper | Code |
|:----:|:-----:|:------|:----:|
| 2024 | arXiv | [Navigation as Attackers Wish? Towards Building Robust Embodied Agents under Federated Learning](https://arxiv.org/abs/2211.14769) | - |
| 2023 | IEEE TPAMI | [Towards Deviation-Robust Agent Navigation via Perturbation-Aware Contrastive Learning](https://doi.org/10.1109/TPAMI.2023.3273594) | - |
| 2023 | arXiv | [VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models](https://arxiv.org/abs/2307.05973) | - |

---

## 🔮 Challenges and Future Directions

We identify key challenges and future research directions for achieving privacy-aware and trustworthy embodied AI:

### 🛡️ Safety-first VLA

Current VLA methods prioritize task performance and generalization, yet lack a comprehensive mechanism to explicitly integrate safety constraints with embodied capabilities.

| Direction | Description | Key References |
|:----------|:------------|:---------------|
| **Automated Safety Frameworks** | Leverage foundation models to detect emerging unsafe patterns (e.g., a home service robot failing to recognize inflammable items near stove flames) | [SafeVLA](https://arxiv.org/abs/2503.03480) |
| **User-defined Physical Safety Boundaries** | Accommodate personalized deployment requirements in embodied tasks | - |

### 🔐 Privacy-aware VLA

Most VLA methods uniformly process visual inputs without distinguishing privacy-sensitive content. Automatic detection of privacy risks is essential for embodied AI tasks.

| Direction | Description | Key References |
|:----------|:------------|:---------------|
| **Adaptive Privacy Protection** | Automatic detection strategies for high-risk elements such as biometrics, medical records, and financial documents | [Privacy Taxonomy](https://arxiv.org/abs/2509.23827), [Context-Aware Framework](https://doi.org/10.3390/s25196105) |
| **User-defined "Red-line Rules"** | Prohibiting perception in private spaces (e.g., bathrooms) that override automatic inference | - |
| **Federated Embodied AI** | Align privacy-aware semantics with embodied performance under "data stays, model moves" paradigm | [FedVLN](https://arxiv.org/abs/2203.14936), [FedVLA](https://arxiv.org/abs/2508.02190) |

### ⚡ On-Device Inference Acceleration

While large VLA methods deliver good success rates, their dependence on cloud deployment introduces privacy risks and compromises real-time responsiveness.

| Direction | Description | Key References |
|:----------|:------------|:---------------|
| **Embodiment-aware Acceleration** | Maintain control fidelity and real-time performance under privacy-aware constraints | [Real-time Action Chunking](https://arxiv.org/abs/2506.07339) |
| **Static Backbone Compression** | Efficient model compression for edge deployment | [TinyVLA](https://arxiv.org/abs/2406.04339), [SmolVLA](https://arxiv.org/abs/2506.01844) |
| **Dynamic Computation Pathways** | Adaptive inference based on task complexity | [DEER-VLA](https://arxiv.org/abs/2410.13383), [MoLe-VLA](https://arxiv.org/abs/2503.20384) |
| **Quantization-aware Training** | Low-bit VLA models for resource-constrained platforms | [BitVLA](https://arxiv.org/abs/2506.07530) |

### ⚖️ Utility-Accuracy-Privacy Trade-off

Current VLA research optimizes utility, efficiency, and privacy separately, hampering systematic cross-system comparisons.

| Direction | Description | Key References |
|:----------|:------------|:---------------|
| **Standardized Protocols** | Model multifaceted trade-offs among task performance, computational cost, and privacy guarantees | [Pareto Frontiers](https://arxiv.org/abs/2302.09183) |
| **Unified Evaluation Frameworks** | Enable Pareto frontier construction across deployment contexts | [Efficiency Survey](https://arxiv.org/abs/2510.24795) |

---

## 🔖 Citation

If you find this survey helpful, please consider citing:

```bibtex
@misc{author2025privacyaware,
  title={Privacy-Aware Embodied AI: A Position Paper},
  author={Author Name},
  year={2025},
  eprint={2510.xxxxx},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2510.xxxxx},
}
```

---

## 📧 Contact

For any questions or suggestions, please feel free to contact us:

<div align="center">

| Author | Email | Affiliation |
|:-------|:------|:------------|
| **Author Name** | email@example.com | Your University |

</div>

---

## 🙏 Acknowledgements

We thank all the authors of the papers included in this survey.

---

<div align="center">

**If you find this repository useful, please give us a ⭐!**

</div>

---

<p align="center">
  <i>Last updated: June 2025</i>
</p>

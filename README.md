<div align="center">

# 🔐 A Survey on Privacy-Preserving Vision-Language-Action Models

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![arXiv](https://img.shields.io/badge/arXiv-2510.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2510.xxxxx)
[![GitHub stars](https://img.shields.io/github/stars/YourUsername/Privacy-VLAs-Survey?style=social)](https://github.com/YourUsername/Privacy-VLAs-Survey)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/YourUsername/Privacy-VLAs-Survey/graphs/commit-activity)

</div>

<div align="center">
  <img src="assets/banner.png" width="100%" alt="Survey Banner"/>
</div>

---

<p align="center">
  <b>🔥 This is a curated paper list of "A Survey on Privacy-Preserving Vision-Language-Action Models".</b>
  <br>
  To the best of our knowledge, this work presents the <b>first comprehensive survey</b> specifically dedicated to privacy-preserving techniques in VLAs. 
  <br>
  We will continue to <b>UPDATE</b> this repository with the latest developments! 
  <br>
  ⭐ Star us to stay tuned! 😘
</p>

---

## 📢 News

| Date | News |
|:-----|:-----|
| 🔥 2025/06 | Our survey paper is released on arXiv! |
| 🔥 2025/06 | Repository created! |

---

## 🔍 Table of Contents

- [📖 Introduction](#-introduction)
- [🏗️ Taxonomy](#️-taxonomy)
- [🛡️ Privacy Threats in VLAs](#️-privacy-threats-in-vlas)
- [🔒 Privacy-Preserving Techniques](#-privacy-preserving-techniques)
  - [Federated Learning](#federated-learning)
  - [Differential Privacy](#differential-privacy)
  - [Secure Aggregation](#secure-aggregation)
  - [Data Anonymization](#data-anonymization)
- [📊 Benchmarks & Datasets](#-benchmarks--datasets)
- [🚀 Applications](#-applications)
- [🔮 Future Directions](#-future-directions)
- [🔖 Citation](#-citation)
- [📧 Contact](#-contact)
- [🙏 Acknowledgements](#-acknowledgements)

---

## 📖 Introduction

<div align="center">
  <img src="assets/overview.png" width="90%" alt="Survey Overview"/>
  <br>
  <em>Fig. 1: Overview of our survey on Privacy-Preserving Vision-Language-Action Models.</em>
</div>

<br>

Vision-Language-Action (VLA) models have emerged as a promising paradigm for embodied AI, enabling robots to understand visual scenes, process natural language instructions, and execute physical actions. However, deploying VLAs in real-world scenarios raises significant **privacy concerns**:

- 🏠 **Environmental Privacy**: Home layouts, personal belongings, sensitive documents
- 👤 **Personal Privacy**: Human faces, activities, behavioral patterns  
- 🔐 **Data Security**: Training data leakage, model inversion attacks
- 📡 **Communication Privacy**: Data transmission vulnerabilities

This survey provides a comprehensive review of privacy-preserving techniques for VLAs, covering the entire lifecycle from data collection to model deployment.

---

## 🏗️ Taxonomy

<div align="center">
  <img src="assets/taxonomy.png" width="95%" alt="Taxonomy"/>
  <br>
  <em>Fig. 2: Taxonomy of Privacy-Preserving Techniques in VLAs.</em>
</div>

---

## 🛡️ Privacy Threats in VLAs

<div align="center">
  <img src="assets/threats.png" width="85%" alt="Privacy Threats"/>
  <br>
  <em>Fig. 3: Privacy Threats in Vision-Language-Action Models.</em>
</div>

<br>

| Threat Category | Description | Attack Examples |
|:----------------|:------------|:----------------|
| **Data Leakage** | Sensitive information exposed during training | Membership inference, data extraction |
| **Model Inversion** | Reconstructing training data from model | Gradient-based reconstruction |
| **Attribute Inference** | Inferring sensitive attributes | Demographic, behavioral inference |
| **Environment Exposure** | Leaking physical environment details | Scene reconstruction, object detection |

---

## 🔒 Privacy-Preserving Techniques

### Federated Learning

<div align="center">
  <img src="assets/federated.png" width="80%" alt="Federated Learning"/>
  <br>
  <em>Fig. 4: Federated Learning Framework for VLAs.</em>
</div>

<br>

| Year | Venue | Paper | Website | Code |
|:----:|:-----:|:------|:-------:|:----:|
| 2025 | arXiv | [FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts](https://arxiv.org/abs/2508.02190) | - | - |
| 2024 | NeurIPS | [Paper Title Here](https://arxiv.org/) | [🌐](https://project-page.com) | [💻](https://github.com/) |
| 2024 | ICML | [Paper Title Here](https://arxiv.org/) | [🌐](https://project-page.com) | [💻](https://github.com/) |

### Differential Privacy

| Year | Venue | Paper | Website | Code |
|:----:|:-----:|:------|:-------:|:----:|
| 2025 | ICLR | [Paper Title Here](https://arxiv.org/) | [🌐](https://project-page.com) | [💻](https://github.com/) |
| 2024 | CVPR | [Paper Title Here](https://arxiv.org/) | [🌐](https://project-page.com) | [💻](https://github.com/) |

### Secure Aggregation

| Year | Venue | Paper | Website | Code |
|:----:|:-----:|:------|:-------:|:----:|
| 2025 | arXiv | [Paper Title Here](https://arxiv.org/) | - | - |
| 2024 | AAAI | [Paper Title Here](https://arxiv.org/) | [🌐](https://project-page.com) | [💻](https://github.com/) |

### Data Anonymization

| Year | Venue | Paper | Website | Code |
|:----:|:-----:|:------|:-------:|:----:|
| 2025 | ICRA | [Paper Title Here](https://arxiv.org/) | [🌐](https://project-page.com) | [💻](https://github.com/) |
| 2024 | CoRL | [Paper Title Here](https://arxiv.org/) | [🌐](https://project-page.com) | [💻](https://github.com/) |

---

## 📊 Benchmarks & Datasets

| Dataset | Year | Scale | Privacy Features | Link |
|:--------|:----:|:-----:|:-----------------|:----:|
| Dataset1 | 2024 | 100K | Anonymized faces | [🔗](https://link.com) |
| Dataset2 | 2024 | 50K | Differential privacy | [🔗](https://link.com) |
| Dataset3 | 2023 | 200K | Federated collection | [🔗](https://link.com) |

---

## 🚀 Applications

<div align="center">
  <img src="assets/applications.png" width="90%" alt="Applications"/>
  <br>
  <em>Fig. 5: Applications of Privacy-Preserving VLAs.</em>
</div>

<br>

| Application Domain | Privacy Requirements | Representative Works |
|:-------------------|:---------------------|:---------------------|
| 🏠 **Home Robotics** | User privacy, environment protection | [Paper1](link), [Paper2](link) |
| 🏥 **Healthcare** | Patient data, medical records | [Paper1](link), [Paper2](link) |
| 🏭 **Industrial** | Trade secrets, process data | [Paper1](link), [Paper2](link) |
| 🚗 **Autonomous Driving** | Location privacy, passenger data | [Paper1](link), [Paper2](link) |

---

## 🔮 Future Directions

We identify several promising research directions:

1. **Scalable Federated VLA Training** - Efficient communication and computation
2. **Privacy-Utility Trade-offs** - Balancing model performance with privacy guarantees
3. **Cross-Modal Privacy** - Protecting privacy across vision, language, and action modalities
4. **Real-time Privacy Protection** - On-device privacy-preserving inference
5. **Regulatory Compliance** - GDPR, CCPA compliance for embodied AI

---

## 🔖 Citation

If you find this survey helpful, please consider citing:

```bibtex
@misc{your2025privacy,
  title={A Survey on Privacy-Preserving Vision-Language-Action Models},
  author={Your Name and Co-author Name and Other Authors},
  year={2025},
  eprint={2510.xxxxx},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2510.xxxxx},
}
```

---

## 📧 Contact

For any questions or suggestions, please feel free to contact us:

<div align="center">

| Author | Email | Affiliation |
|:-------|:------|:------------|
| **Your Name** | your.email@example.com | Beijing Jiaotong University |
| **Co-author** | coauthor@example.com | University Name |

</div>

---

## 🙏 Acknowledgements

We thank all the authors of the papers included in this survey. Special thanks to:
- [Related Survey 1](link)
- [Related Survey 2](link)
- [Awesome VLA Repository](link)

---

<div align="center">

**If you find this repository useful, please give us a ⭐!**

[![Star History Chart](https://api.star-history.com/svg?repos=YourUsername/Privacy-VLAs-Survey&type=Date)](https://star-history.com/#YourUsername/Privacy-VLAs-Survey&Date)

</div>

---

<p align="center">
  <i>Last updated: June 2025</i>
</p>

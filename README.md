<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src=https://i.pinimg.com/originals/04/de/f7/04def7755255edefe5cd22ef37e9f41c.jpg alt="Project logo"></a>
</p>

<h3 align="center">PaDiM: An Unsupervised Learning Framework for Anomaly Detection in Medical Images </h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Acknowledgements](#acknowledgements)
- [References](#authors)
- [Authors](#authors)

## 🧐 About

This repository contains the implementation of PaDiM (Patch Distribution Modeling), an unsupervised learning algorithm for anomaly detection in images. This project is based on a comparative study of unsupervised learning paradigms, focusing on PaDiM and Patched Diffusion Models. Our study explores the effectiveness of PaDiM in detecting anomalies with high precision and low computational cost, making it suitable for a wide range of applications from industrial inspection to medical diagnosis.
For more information, please follow the link below. 

### [Constrating Unsupervised Learning Paradigms: A Comparative Study of PaDiM and Patched Difussion Models in MRI Anomaly Detection](https://amber-gram-6b4.notion.site/Constrating-Unsupervised-Learning-Paradigms-A-Comparative-Study-of-PaDiM-and-Patched-Difussion-Mode-3e70a0e097444900a05a07ebaafd909e)

## 🏁 Getting Started 

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them.

```bash
git clone <repository-url>
cd Unsupervised-Anomaly-Detection-in-Medical-Images-via-PaDiM
pip install -r requirements.txt
```

### Dataset
[FAST_MRI and IXI](https://syncandshare.lrz.de/dl/fiH6r4B6WyzAaxZXTEAYCE/data.zip) dataset used in this project. Follow the link to download it to your local.

* Download the data.
* Unzip the data and move it to cloned repository folder

```bash
cd Unsupervised-Anomaly-Detection-in-Medical-Images-via-PaDiM
wget -O data.zip https://syncandshare.lrz.de/dl/fiH6r4B6WyzAaxZXTEAYCE/data.zip
unzip data.zip
rm -rf data.zip
```
### Update config.yaml
Update the config file based on your model choice.

#### resnet18
```bash
  backbone : 'resnet18' 
  target_dimension : 448
  output_dimension : 180 
```
#### wide_resnet50_2
```bash
  backbone : 'wide_resnet50_2' 
  target_dimension : 1792
  output_dimension : 550 
```


## Usage

To run PaDiM on your dataset, follow the steps outlined in the main Jupyter notebook (padim.ipynb). The notebook provides a comprehensive guide to training the PaDiM model with your data, performing anomaly detection, and evaluating the model's performance.

```bash
jupyter notebook padim.ipynb
```

## Methodology
![alt text](https://miro.medium.com/v2/resize:fit:1400/1*nETEVc6NilhFnXzxcr9RgA.png)
PaDiM combines the strengths of patch-based methods and distribution modeling to detect anomalies in an unsupervised manner. It leverages deep feature representations from pre-trained CNN models and models the distribution of normal patches in a feature space to identify deviations indicative of anomalies. For a detailed explanation of the methodology and comparative analysis with Patched Diffusion Models, refer to our [comparative study article](https://amber-gram-6b4.notion.site/Constrating-Unsupervised-Learning-Paradigms-A-Comparative-Study-of-PaDiM-and-Patched-Difussion-Mode-3e70a0e097444900a05a07ebaafd909e)

## Results
Below is the implementation results of the test set (ROCAUC & AUPRC) on the [FAST_MRI and IXI](https://syncandshare.lrz.de/dl/fiH6r4B6WyzAaxZXTEAYCE/data.zip) dataset

                        |       Model        |    R18-Rd100    |    WR50-Rd550    |
                        |--------------------|-----------------|------------------|
                        | absent_septum      | (0.810, 0.488)  | (0.810, 0.456)   |
                        | artefacts          | (0.697, 0.246)  | (0.713, 0.363)   |
                        | craniatomy         | (0.593, 0.451)  | (0.587, 0.447)   |
                        | dural              | (0.769, 0.509)  | (0.740, 0.476)   |
                        | ea_mass            | (0.500, 0.461)  | (0.523, 0.479)   |
                        | edema              | (0.723, 0.469)  | (0.727, 0.524)   |
                        | encephalomalacia   | (0.956, 0.586)  | (0.950, 0.596)   |
                        | enlarged_ventricles| (0.849, 0.602)  | (0.834, 0.601)   |
                        | intraventricular   | (0.979, 0.539)  | (0.967, 0.490)   |
                        | lesions            | (0.853, 0.278)  | (0.845, 0.242)   |
                        | mass               | (0.808, 0.218)  | (0.778, 0.237)   |
                        | posttreatment      | (0.637, 0.419)  | (0.609, 0.129)   |
                        | resection          | (0.834, 0.200)  | (0.800, 0.262)   |
                        | sinus              | (0.764, 0.426)  | (0.793, 0.469)   |
                        | wml                | (0.810, 0.296)  | (0.805, 0.354)   |
                        | other              | (0.641, 0.333)  | (0.580, 0.365)   |

### Qualitatives
![Result Image 1](https://github.com/ardamamur/Unsupervised-Anomaly-Detection-in-Medical-Images-via-PaDiM/sources/artefacts_5.png?raw=true)

![Result Image 2](https://github.com/ardamamur/Unsupervised-Anomaly-Detection-in-Medical-Images-via-PaDiM/sources/mass_13.png?raw=true)

![Result Image 3](https://github.com/ardamamur/Unsupervised-Anomaly-Detection-in-Medical-Images-via-PaDiM/sources/mass_4.png?raw=true)

![Result Image 4](https://github.com/ardamamur/Unsupervised-Anomaly-Detection-in-Medical-Images-via-PaDiM/sources/lesions_13.png?raw=true)



## Acknowledgments

This project incorporates code from the following open-source repositories:

- PaDiM Anomaly Detection and Localization by Xiahaifeng1995
  [GitHub Repository](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)

- Machine Anomaly Detection Seminar at compai-lab for the Winter Semester 2023
  [GitHub Repository](https://github.com/compai-lab/mad_seminar_ws23/)

- Anomalib: A Deep Learning Library for Anomaly Detection by OpenVINO™ Toolkit
  [GitHub Repository](https://github.com/openvinotoolkit/anomalib)

## References

[1] Defard, Thomas, et al. "Padim: a patch distribution modeling framework for anomaly detection and localization."International Conference on Pattern Recognition. Cham: Springer International Publishing, 2021.


## Authors
* [Arda Mamur](https://github.com/ardamamur)

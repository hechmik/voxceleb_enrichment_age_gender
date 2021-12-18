# VoxCeleb enrichment for Age and Gender recognition

This repository contains all the material related to the paper "VoxCeleb enrichment for Age and Gender recognition" submitted for publication at [ASRU 2021](https://asru2021.org/). For those mainly interested in downloading data you can download the [**ENRICHED DATASET**](dataset/final_dataframe_extended.csv) csv file.
## Arxiv Link
https://arxiv.org/abs/2109.13510
## Abstract

VoxCeleb datasets are widely used in speaker recognition studies. Our work serves two purposes.

First, we provide speaker age labels and (an alternative) annotation of speaker gender. 
  Second, we demonstrate the use of this metadata by constructing age and gender recognition models with different features and classifiers. We query different celebrity databases and apply consensus rules to derive age and gender labels. We also compare the original VoxCeleb gender labels with our labels to identify records that might be mislabeled in the original VoxCeleb data.
  
On modeling side, the lowest mean absolute error (MAE) in age regression, 9.443 years, is obtained using i-vector features with ridge regression. This indicates challenge in age estimation from in-the-wild style speech data.

## Authors
- [Khaled Hechmi](https://www.linkedin.com/in/hechmikhaled/)
- [Trung Ngo Trong](https://scholar.google.com/citations?user=EZEq2nAAAAAJ&hl=it&oi=ao)
- [Ville Hautamäki](https://scholar.google.com/citations?user=esQWyTcAAAAJ&hl=it)
- [Tomi Kinnunen](https://scholar.google.com/citations?user=e3SPjpoAAAAJ&hl=it)

## Repo structure
This repository is structured as follows:
- [dataset](dataset/): here the [**ENRICHED DATASET**](dataset/final_dataframe_extended.csv) can be found and downloaded, as well as support files detailing which records have been used for training and testing
- [best_models](best_models/): the best models reported in the paper, Linear Regression with i-Vectors (Age regression) and Logistic regression with i-Vectors (Gender recognition), are made available so that other users can try them in a variety of scenarios (assuming that features where computed as described)
- [notebooks](notebooks/): Python scripts and Jupyter notebooks used throughout the various steps

## Aknowledgments
This work has been partially sponsored by [Academy of Finland](https://www.aka.fi/en) (proj. no. 309629). 

Considering the nature of the work, we would like to cite also in this README the original [VoxCeleb 1 and VoxCeleb 2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) papers:
```
[1] A. Nagrani*, J. S. Chung*, A. Zisserman, VoxCeleb: a large-scale speaker identification dataset, 
INTERSPEECH, 2017

[2] J. S. Chung*, A. Nagrani*, A. Zisserman, VoxCeleb2: Deep Speaker Recognition, 
INTERSPEECH, 2018
```

## Similar works
This work was carried out in 2020 when the first author was affiliated with University of Eastern Finland. The authors came later across an independent but [closely related work](https://ieeexplore.ieee.org/document/9414272) that addresses age labeling of VoxCeleb. The key difference between our work and theirs is that we assigned age labels based on the videos semantic and people identity, while they trained a facial age estimation model for the labeling task, taking as input the visual frames of the original YouTube videos. For readers convenience here it follows the paper's full reference, together with their github repo.
```
N. Tawara, A. Ogawa, Y. Kitagishi and H. Kamiyama, "Age-VOX-Celeb: Multi-Modal Corpus for Facial and Speech Estimation," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 6963-6967, doi: 10.1109/ICASSP39728.2021.9414272.
```
**GITHUB Repository**: https://github.com/nttcslab-sp/agevoxceleb
## Contact information

For any comment, clarification or suggestion please feel free to open an issue here in GitHub and/or send me an email at **hechmi DOT khaled1995 AT gmail DOT com** 

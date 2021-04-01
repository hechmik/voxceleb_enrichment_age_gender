# VoxCeleb enrichment for Age and Gender recognition

This repository contains all the material related to the paper "VoxCeleb enrichment for Age and Gender recognition" submitted for publication at Interspeech 2021.

## Abstract

VoxCeleb datasets are widely used in speaker recognition studies. Our work serves two purposes.

First, we provide speaker age labels and (an alternative) annotation of speaker gender. 
  Second, we demonstrate the use of this metadata by constructing age and gender recognition models with different features and classifiers. We query different celebrity databases and apply consensus rules to derive age and gender labels. We also compare the original VoxCeleb gender labels with our labels to identify records that might be mislabeled in the original VoxCeleb data.
  
On modeling side, the lowest mean absolute error (MAE) in age regression, 9.443 years, is obtained using i-vector features with ridge regression. This indicates challenge in age estimation from in-the-wild style speech data.

## Authors
- Khaled Hechmi
- Trung Ngo Trong
- Ville Hautamäki
- Tomi Kinnunen

## Repo structure
This repository is structured as follows:
- [dataset](dataset/): here the [enriched dataset](final_dataframe_extended.csv) can be found, as well as support files detailing which records have been used for training and testing
- [best_models](best_models/): the best models reported in the paper, Linear Regression with i-Vectors (Age regression) and Logistic regression with i-Vectors (Gender recognition), are made available so that other users can try them in a variety of scenarios (assuming that
- [notebooks](notebooks/): Python scripts and Jupyter notebooks used throughout the various steps

## Aknowledgments
This work has been partially sponsored by [Academy of Finland](https://www.aka.fi/en) (proj. no. 309629). 

Considering the nature of the work, we would like to cite also in this README the original VoxCeleb 1 and VoxCeleb 2 papers:
```
[1] A. Nagrani*, J. S. Chung*, A. Zisserman, VoxCeleb: a large-scale speaker identification dataset, 
INTERSPEECH, 2017

[2] J. S. Chung*, A. Nagrani*, A. Zisserman, VoxCeleb2: Deep Speaker Recognition, 
INTERSPEECH, 2018
```

## Contact information

For any comment, clarification or suggestion please feel free to open an issue here in GitHub and/or send me an email at **hechmi DOT khaled1995 AT gmail DOT com** 

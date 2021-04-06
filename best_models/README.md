In this folder you can find the three best models as reported in the original paper. Here it follows a brief description on their main characteristics and how to load them in your scripts/notebooks.
# age_regression-mfcc

This a 1-dimensional Convolutional Neural Network trained for the age regression task in [this notebook](https://github.com/hechmik/voxceleb_enrichment_age_gender/blob/main/notebooks/03.3-Age%20regression-Train_test-MFCC.ipynb): as input features it uses MFCC compute as described in [Section 02 of this README file](https://github.com/hechmik/voxceleb_enrichment_age_gender/tree/main/notebooks).

This models has been traind with Keras, therefore you can load them with the following instructions:
```python
from tensorflow import keras
model = keras.models.load_model('age_regression-mfcc/')
```
**MAE: 9.443**
# age_regression-ivec-lm_unbalanced.pkl

This model is a classical Linear Regression model that takes as input i-Vectors. This model has been trained with Scikit-learn and can be loaded as follows:
```python
import pickle
with open('age_regression-ivec-lm_unbalanced.pkl', 'rb') as f:
  model = pickle.load(f)
```
**MAE: 9.443**

# ivec_log_reg_model.torch 

This model, instead, is the one that obtained the highest F1-Score in the gender recognition task. It is a logistic regression model implemented in PyTorch, therefore you can load it with the following instructions:
```
from src.gender_classifiers import LogisticRegression
import torch
model = LogisticRegression(512, 1)
model.load_state_dict(torch.load('ivec_log_reg_model.torch'))
```
IMPORTANT: You will need to import [this custom file](https://github.com/hechmik/voxceleb_enrichment_age_gender/blob/main/notebooks/src/gender_classifiers.py) containing all the models used for predicting gender.

**F1-Score: 0.9829**

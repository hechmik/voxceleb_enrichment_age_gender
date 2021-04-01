In this folder a number of scripts and notebooks are reported. Here it follows a general overview:
### Disclaimer
From a software engineering it is clear that there is large room of improvements, as various functions appear as duplicated due to the fact that . The main goal is to enable the reader to understand how things were done and allow as much as possible the reproducibility of all the steps.
If some imports are broken or you don't understand some steps please open an issue and let me know.
## 00 Conda environment
Conda was used for managing the various libraries needed by scripts and notebooks. In particular:
- [environment_enrichment.yml]()
- [environment_asvtorch.yml](./environment_asvtorch.yml): Environment used for computing x- and i-Vectors + training gender recognition models
- [environment_asvtorch.yml](./environment_tensorflow.yml): Environment used in training Keras and Scikit-learning models + all the other activities
The general instruction for installing the various environments is the following:
```conda env create -f environment.yml```
## 01 Enrichment
The enrichment activity was performed using ["01-Enrich_VoxCeleb_Dataset.ipynb"](01-Enrich_VoxCeleb_Dataset.ipynb) Jupyter notebook. This code takes advantage of some.

**IMPORTANT**: It is necessary to have specific credentials for querying Google Knowledge Graph. Detailed instruction can be found in this webpage [https://developers.google.com/knowledge-graph/prereqs](https://developers.google.com/knowledge-graph/prereqs)

## 02 Computation of MFCC, i-Vectors and x-Vectors
This activity was done using [ASVTorch code](https://gitlab.com/ville.vestman/asvtorch) made by Ville Vestman, who have well supported me during this crucial activity.

The repo was basically used as is, apart from the following modifications done in [ivector/run.py](https://gitlab.com/ville.vestman/asvtorch/-/blob/master/asvtorch/recipes/voxceleb/ivector/run.py) and [xvector/run.py](https://gitlab.com/ville.vestman/asvtorch/-/blob/master/asvtorch/recipes/voxceleb/xvector/run.py)
- All the training parts are computed on VoxCeleb1, while the trial data is the whole VoxCeleb 2 corpus. In order to do this please modify all the strings passed as argument to chooseAll() functions
- All steps, except the lasting one, are performed, without altering their sequences. More info about how to call the run.py scripts can be found in the [i-Vector](https://gitlab.com/ville.vestman/asvtorch/-/tree/master/asvtorch/recipes/voxceleb/ivector) and [x-Vector](https://gitlab.com/ville.vestman/asvtorch/-/tree/master/asvtorch/recipes/voxceleb/ivector) README files.

### x-Vector
For the x-Vector training part the following parameters reported in the [run_configs.py](https://gitlab.com/ville.vestman/asvtorch/-/blob/master/asvtorch/recipes/voxceleb/xvector/configs/run_configs.py) file were changed (because of time constraints):
- network.utts_per_speaker_in_epoch = 200
- network.max_epochs = 500

Also the loading part of stage 5 (the effective training of the model) has been modified as follows:
``` python
training_data = UtteranceSelector().choose_all('voxceleb1_combined') # combined = augmented version
training_data.remove_short_utterances(550)  # Remove utts with less than 500 frames
training_data.remove_speakers_with_few_utterances(10)  # Remove spks with less than 10 utts

print('Selecting PLDA training data...')
plda_data = UtteranceSelector().choose_all('voxceleb1_combined')
plda_data.select_random_speakers(400)

trial_data = UtteranceSelector().choose_all('voxceleb1')
trial_data.select_random_speakers(100)
```
## 03 Model training and evaluation
Notebooks starting with "02" and "03" have been used for training and evaluation all the reported models. In  the src folder you can find the methods that effectively build the predictive models according to the specified params + some methods that were tried during the experimentation phase, done using K-Fold CV on the train set, but weren't used in the final train due to their lower results.

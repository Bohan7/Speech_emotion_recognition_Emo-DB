# Speech emotion recognition for EmoDB

- Bohan Wang:
contact: wangbohan307@gmail.com

## Datasets 

1. EmoDB [[1]](#1): The EMODB database is the freely available German emotional database. The database is created by the Institute of Communication Science, Technical University, Berlin, Germany. 
Ten professional speakers (five males and five females) participated in data recording. The database contains a total of 535 utterances. 
The EMODB database comprises of seven emotions: 1) anger; 2) boredom; 3) anxiety; 4) happiness; 5) sadness; 6) disgust; and 7) neutral. 
The data was recorded at a 48-kHz sampling rate and then down-sampled to 16-kHz.

## Models

CNN: Implement 1D convolution on the 1D speech signal.

## Notes
The packages used in the project can be installed using:

``pip install -r requirements.txt``


## Structure
**train_features.npy:** contains the features of the train samples and the augmentated train samples.

**train_y.npy:** contains the lables of the train samples and the augmentated train samples.

**test_features.npy:** contains the features of the test samples.

**test_y.npy** contains the labels of the test samples.

**requirements.txt** contains the packages used for this project.

**models.py:** contains the model definition code of CNN.

**utils.ipynb:** contains the helper functions for pre-processing, data augmentation, train and evaluation.

**Report_SER.ipynb:** contains the code to reproduce the training and visualization. 

## Instuctions

You can see the Exploratory Data Analysis, do cross validation and train the CNN model in:

``Report_SER.ipynb``

## References
<a id="1">[1]</a> 
Burkhardt F., Paeschke A., Rolfes M., Sendlmeier W., Weiss B. 
A database of german emotional speech; Proceedings of the Interspeech; 
Lisbon, Portugal. 4–8 September 2005; pp. 1517–1520.





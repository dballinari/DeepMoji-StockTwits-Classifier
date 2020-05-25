# Fine tuning of DeepMoji on StockTwits messages
## Introduction
This projects fine-tunes the [DeepMoji classifier](https://github.com/bfelbo/DeepMoji) 
using the self-reported sentiments of StockTwits messages. 
## Set-up
The project builds upon the code of Felbo et. al. which can be found 
[here](https://github.com/bfelbo/DeepMoji). Detail instructions for 
setting up their model can be found directly at their GitHub repository. 

For this partiuclar project, using Miniconda with Python 3.7, a virtual 
enviroment of Python 2.7 was set-up:
```bash
conda create -n deepmoji python=2.7
```
In this virtual enviroment, install a 'theano'. Then navigate to the
root folder where the DeepMoji files are located and run the installation
file:
```bash
pip install -e .
```
Next the backend of keras has to be changed: either change in '.keras/keras.JSON' the 
backend from "tensorflow" to "theano" or create a new folder '.keras_python27' (as a 
copy of '.keras/') and change there the file 'keras.JSON'. In latter case, go to 
'_path_\Miniconda3\envs\deepmoji\Lib\site-packages\keras\backend\__init__.py' and change
the location of the keras setting file to point to the newly created folder. Finally, 
if when running the scripts an error about the 'ifelse' function appears, in the file 
where the error occurs add at the beggining 'from theno import ifelse' and replace in the 
code 'theano.ifelse' with 'ifelse'.

## Project structure
The projects consist of mainly four folders:
* data: this folder contains only the script that converts the StockTwits data to a dictionary
with the structure info, texts, val_id, test_id, train_id and saves them as 'pickle'-file. The  
actual data is stored locally on the machine. The folder has the following 
structure:
    * 00_raw: the raw data and help-data
    * 01_processed: the output of the sentiment prediction, i.e. the daily sentiment
    * 02_models: the estimated models saved as `JSON` files
    * 03_model_output: csv-file with the model performance (accuracy, F1-score)
* deepmoji: files as in the original [DeepMoji repository](https://github.com/bfelbo/DeepMoji)
* scripts: contains the scripts used to run the estimation and prediction
* model: files as in the original [DeepMoji repository](https://github.com/bfelbo/DeepMoji)

## Estimation details 
The script 'fit_stocktwits.py' fine-tunes the pre-trained DeepMoji model of 
[Felbo et. al. (2017](https://www.aclweb.org/anthology/D17-1169.pdf) on the self-reported bullishness
or bearishness of StockTwits messages. The models are estimated using StockTwits messages mentioning 
one of 360 US stocks of the S&P 500. For the estimation only messages from 2013-06-01 to 2014-08-31 
are used; this matches the data estimation window used by [Renault (2017)](https://www.sciencedirect.com/science/article/abs/pii/S0378426617301589).
An undersampling technique is then used to create a balanced train data set. 

The fine-tuning on StockTwits messages starts by increasing the number of words in the embedding vocabulary. 
Then, the output layer of the neural network is replaced to account for the fact that we have only two labels
in the StockTwits sentiment classification task. The fine tuning is then done by using the 'chain-saw' approach
suggested by [Felbo et. al. (2017)](https://www.aclweb.org/anthology/D17-1169.pdf).
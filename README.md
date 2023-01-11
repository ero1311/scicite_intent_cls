# Citation Intent Classification on SciCite Dataset

## Installation

Conda environment (recommended)

``` shell
#create the conda environment from the environment.yml file
conda env create -f environment.yml
```

## Training

Data preparation

First we will need to process the citation sentences to get a dataset of embedding vectors. This repo utilizes GloVE and SciBert embeddings of dimensions 300 and 768 respectively. 

``` shell
python preprocess_dataset.py
```

Then having prepared the data you can proceed to training by:
``` shell
python train.py 
```
Before running a training examine the hyperparams carefully to decide whether to change them or leave as they are

## Inference
You can evaluate the performance of any model trained under the ./logs directory. You can find the weights of the best model here: [`best.pth`](https://drive.google.com/file/d/1U2TxYr5w-0ABWFEoIJvqD7hrIIfvwbj3/view?usp=share_link). You can also view the test dataframe with predictions under the label_pred column in the test_predictions.csv file.
``` shell
python eval.py --exp_name <NAME OF THE EXPERIMENT> --emb_type <TYPE OF THE EMBEDDING>
```


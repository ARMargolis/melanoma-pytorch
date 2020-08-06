Goal: A PyTorch model for https://www.kaggle.com/c/siim-isic-melanoma-classification/data
Technique: Pruning https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#iterative-pruning
Theory: Lottery Ticket Hypothesis https://arxiv.org/pdf/1803.03635.pdf


# Setup
## Colab stuff
To use the kaggle API in CoLab, follow the instructions here: [ https://medium.com/@move37timm/using-kaggle-api-for-google-colaboratory-d18645f93648#:~:text=To%20use%20the%20Kaggle%20API,of%20a%20file%20called%20'kaggle ]
(some of the steps are present in the `setting_up_kaggle_credentials.ipynb` notebook, so you can copy them over)

## Dependencies
We will use poetry to manage dependencies, so all you have to do is:
1. Clone the repo
2. Create a virtualenv in the repo root with `virtualenv .venv`, then run `source .venv/bin/activate`
3. Run `pip install poetry`, then `poetry install` and all the dependencies should install

# Directories
data:
-labels: csv files containing the labels and metadata of the images (train.csv, test.csv)
-jpeg: the jpeg folder from the kaggle dataset (we will operate on the jpeg)
--train: training images
--test: testing images
model: model architechtures, utils, torch.Datasets/DataLoaders
scripts: training scripts
notebooks: place to store notebooks and experiments
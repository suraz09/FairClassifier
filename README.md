# FairClassifier

**Machine Learning** models are extensively used for the purpose of determining access to services such as eligibility for loans, insurance etc. Despite gains in efficiency of these models, unintentional discrimination flaw of the model has not been fully addressed. 

The Fair Classifier is an open source package that evaluate the fairness of a Neural Network. The goal is to build a command-line playground where user can check the fairness of the model by removing different sensitive attributes from the training data.
 


### Fairness metrics implemented
> The fairness contraint P% rule is based on [this paper.](https://arxiv.org/pdf/1507.05259.pdf)

### Biasness mitigation implemented
> A neural network is attached with an adversarial network to help mitigate the bias. For further reading please follow this [Link.](https://blog.godatadriven.com/fairness-in-ml)


## Setup
Follow these steps for installation.

### Manual Installation
Clone this repository.

`git clone git@github.com:suraz09/FairClassifier.git`

To run the example script, install the additional libraries specified in requirements.txt file as follows:

`pip install -r requirements.txt`

Then change your directory

`cd FairClassifier`

Currently, this package works only with 2 sensitive attribute of the dataset i.e race and sex.

To check the fairness of the model for race attribute run as:

`python src/notebooks/main.py race African-American`

To check the fairness of the model on sex attribute run as:

`python src/notebooks/main.py sex Female`




## Motivation
This project is inspired by this [Blog.](https://blog.godatadriven.com/fairness-in-ml) 


## Built with
`python`

`jupyter-notebook`

`AWS`

## DataSet
Compas Project [Dataset](https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv)

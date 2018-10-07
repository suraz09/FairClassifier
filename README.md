# FairClassifier

**Machine Learning** models are extensively used for the purpose of determining access to services such as credit, insurance, and employment. Despite gains in efficiency of these models, unintentional discrimination flaw of the model has not been fully addressed. 

The Fair Classifier is an open source package that evaluates the fairness of a Neural Network based on the data it has been trained on.
 
This python package includes a set of metrics to test models for biasness and use adversarial network to help mitigate that biasness of the training data set.

This library is still in development. We encourage the contribution of other metrics, dataset and approach to mitigate the biasness.

Fairness metrics implemented
> The fairness contraint P% rule is based on this paper. [Link](https://arxiv.org/pdf/1507.05259.pdf)

Biasness mitigation implemented
> A neural network is attached with an adversarial network to help mitigate the bias. For further reading please follow [Link](https://blog.godatadriven.com/fairness-in-ml)


# Setup
Installation on a Unix system is easy. Just follow these steps

### Manual Installation
Clone this repository.

`git clone git@github.com:suraz09/FairNN.git`

To run the example script, install the additional libraries specified in requirements.txt file as follows:

`pip install -r requirements.txt`

Then change your directory

`cd FairClassifier`

`python src/notebooks/main.py`


## Motivation
This project is inspired by this [Blog.](https://blog.godatadriven.com/fairness-in-ml) 


## Built with
`python`

`jupyter-notebook`

`AWS`

## DataSet
Compas Project [Dataset](https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv)

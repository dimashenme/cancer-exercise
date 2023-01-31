This script is an exercise to apply neural networks to a
classification problem using PyTorch. The dataset (GSE6008, taken from
https://sbcb.inf.ufrgs.br/cumida) contains genetic markers (>20000)
associated with particular types of ovarian cancer. The script applies
PCA to reduce the number of features and then trains a neural network
with two hidden layers in order to predict the cancer type. It uses
tensorboard to log the losses during the training and then computes
the accuracy of the network's productions on the test data.

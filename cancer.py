import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import sklearn.metrics
import sklearn.decomposition as dec
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter


# build a 1-layer classification neural network

class ClassificationNet(nn.Module):
    def __init__(self, ninputs, nhidden, nhidden2, noutputs, actfn=torch.sigmoid):    
        super(ClassificationNet,self).__init__()
        self.layer1 = nn.Linear(ninputs, nhidden)
        self.layer2 = nn.Linear(nhidden, nhidden2)
        self.layer3 = nn.Linear(nhidden2, noutputs)
        self.actfn = actfn

    def forward(self,input):
        x=self.actfn(self.layer1(input))
        x=self.actfn(self.layer2(x))
        x=self.actfn(self.layer3(x))
        return nn.Softmax(dim=1)(x)           

def training_loop(desc, network, dataset):    
    LEARNING_RATE = 3e-4
    NEPOCHS = 3000

    dloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
#    dloader = DataLoader(dataset, batch_size=20, shuffle=True, drop_last=True)

    lossf = nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(network.parameters(),
                               lr=LEARNING_RATE)
    #losses = []
    
    for epoch in range(NEPOCHS):
        for features, labels in dloader:
            optimizer.zero_grad()
            prediction = network(features)
            loss = lossf(prediction, labels)
            writer.add_scalar(desc, loss, epoch)
            #losses.append(loss.item())
            loss.backward()
            optimizer.step()       

    #plt.plot(np.linspace(0, len(losses), len(losses)), np.array(losses))
    #plt.yscale('log')
    #plt.show()

    writer.flush()
        

def plot_data(data, labels, nlabels):
    plt.clf()

    for n in range(nlabels):
        indices = np.where(labels==n)
        this_label = data[indices]        
        plt.scatter(this_label[:,1], this_label[:,2])

    plt.show()

def plot_data_projection(data, labels, nlabels):
    meta = np.asarray([f"Type {a}" for a in labels])
    
    writer.add_embedding(data[:, 1:],
                         metadata=meta,
                         tag="Cancer types")

    writer.flush()

    
def eval_predictions(network, features, labels):
    with torch.no_grad():
        prediction = torch.max(network(torch.Tensor(features)), dim=1)[1]

    print("Accuracy: ", sklearn.metrics.accuracy_score(prediction.numpy(), labels))

    
def do_real_data():

    df_train = pd.read_csv("train.csv")
    df_train.fillna(0, inplace=True)
    
    labels = set()
    for a in df_train['type']:
        labels.add(a)

    labels_list = list(labels)
    nlabels = len(labels_list)
    labels_dict = {}

    i=0
    for t in labels_list:
        labels_dict[t]=i
        i += 1

    # PCA for dimension reduction        

    pca = dec.PCA(n_components=0.99, svd_solver='full')
    pca_eigenvectors = pca.fit_transform(df_train.iloc[:, 3:])
    n_pca_features = pca_eigenvectors.shape[1]
    
    # make a numpy array with 0-th column labels
    
    labels_column = np.asarray([labels_dict[l] for l in df_train.iloc[:, 2]])

    # make a plot
    
    #plot_data(pca_eigenvectors, labels_column, nlabels)
    plot_data_projection(pca_eigenvectors, df_train['type'], nlabels)

    train_dataset = TensorDataset(
        torch.tensor(pca_eigenvectors, dtype=torch.float),
        torch.tensor(labels_column, dtype=torch.long))
    
    network = ClassificationNet(n_pca_features,
                                int(n_pca_features/8),
                                int(n_pca_features/4),
                                nlabels)
    training_loop("Train data losses", network, train_dataset)

    # check prediction power of the trained network on the test data

    print("Evalating accuracy on training data:\n")
    eval_predictions(network, pca_eigenvectors, labels_column)

    # check prediction power of the trained network on the test data

    df_test = pd.read_csv('test.csv')
    df_test.fillna(0, inplace=True)

    # reusing  pca instance from before
    test_pca_eigenvectors = pca.transform(df_test.iloc[:, 3:])
    test_labels_column = np.asarray([labels_dict[l] for l in df_test.iloc[:, 2]])
    
    test_dataset = TensorDataset(
        torch.tensor(test_pca_eigenvectors, dtype=torch.float),
        torch.tensor(test_labels_column, dtype=torch.long))

    # run the training loop once more on the test data
    # to look at the losses
    
    network2 = ClassificationNet(n_pca_features,
                                int(n_pca_features/8),
                                int(n_pca_features/4),
                                nlabels)
    training_loop("Test data losses", network2, test_dataset)


    print("Evalating accuracy on test data:\n")
    eval_predictions(network, test_pca_eigenvectors, test_labels_column)

def do_random_data():    
    ndim = 2
    nlabels=2
    nfeatures = 1000
    vectors = np.random.rand(nfeatures, ndim)

    labels_list = []
    for p in vectors:
        labels_list.append(float(bool(np.sum(p**2) < 1)))
    labels = np.asarray(labels_list)

    # do train/test split here
    global train_vectors, test_vectors, train_labels, test_labels
    train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors, labels)

    train_dataset = TensorDataset(
        torch.tensor(train_vectors, dtype=torch.float),
        torch.tensor(train_labels, dtype=torch.long))

    test_dataset = TensorDataset(
        torch.tensor(test_vectors, dtype=torch.float),
        torch.tensor(test_labels, dtype=torch.long))
    

    network = ClassificationNet(ndim, 10, 10, nlabels, actfn=torch.relu)       
    training_loop("Train data losses", network, train_dataset)

    eval_predictions(network, test_vectors, test_labels)

    network2 = ClassificationNet(ndim, 10, 10, nlabels, actfn=torch.relu)       
    training_loop("Test data losses", network2, test_dataset)
    
    

writer = SummaryWriter()

#do_real_data()
do_random_data()

    
# # Eduard B, [1/25/23 2:29 PM]
# # 6. Сделать display tools с помощью tensorboard

# # Eduard B, [1/25/23 2:30 PM]
# # from torch.utils.tensorboard import SummaryWriter

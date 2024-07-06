# -*- coding: utf-8 -*-
"""tutorial_variational_classifier.ipynb
modified from: https://pennylane.ai/qml/demos/tutorial_variational_classifier/
"""

#pip install pennylane

# Commented out IPython magic to ensure Python compatibility.
# This cell is added by sphinx-gallery
# It can be customized to whatever you like
# %matplotlib inline

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

"""Quantum and classical nodes
===========================

We then create a quantum device that will run our circuits.

"""

dev = qml.device("default.qubit")

"""Variational classifiers usually define a "layer" or "block", which is an
elementary circuit architecture that gets repeated to build the full
variational circuit.

Our circuit layer will use four qubits, or wires, and consists of an
arbitrary rotation on every qubit, as well as a ring of CNOTs that
entangles each qubit with its neighbour. Borrowing from machine
learning, we call the parameters of the layer `weights`.

"""

def layer(layer_weights):
    for wire in range(4):
        qml.Rot(*layer_weights[wire], wires=wire)

    for wires in ([0, 1], [1, 2], [2, 3], [3, 0]):
        qml.CNOT(wires)

def state_preparation(x):
    qml.AngleEmbedding(x, wires=[0, 1,2,3])

"""Now we define the variational quantum circuit as this state preparation
routine, followed by a repetition of the layer structure.

"""

@qml.qnode(dev)
def circuit(weights, x):
    state_preparation(x)

    for layer_weights in weights:
        layer(layer_weights)

    return qml.expval(qml.PauliZ(0))

"""If we want to add a "classical" bias parameter, the variational quantum
classifier also needs some post-processing. We define the full model as
a sum of the output of the quantum circuit, plus the trainable bias.

"""

def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

"""Cost
====

In supervised learning, the cost function is usually the sum of a loss
function and a regularizer. We restrict ourselves to the standard square
loss that measures the distance between target labels and model
predictions.

"""

def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return np.mean((labels - qml.math.stack(predictions)) ** 2)

"""To monitor how many inputs the current classifier predicted correctly,
we also define the accuracy, or the proportion of predictions that agree
with a set of target labels.

"""

def accuracy(labels, predictions):
    acc = np.sum(labels*predictions>0)
    acc = acc / len(labels)
    return acc

"""For learning tasks, the cost depends on the data - here the features and
labels considered in the iteration of the optimization routine.

"""

def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)

x1=np.random.normal(loc=[-4,-2,0,2],size=(1000,4))
x2=np.random.normal(loc=[-3,-1,1,3],size=(1000,4))
from sklearn.preprocessing import MinMaxScaler
X=np.concatenate((x1,x2),axis=0)
print (np.mean(X,axis=0))
scaler = MinMaxScaler((0,np.pi))
scaler.fit(X)
X=scaler.transform(X)
print (np.mean(X,axis=0))
print (x1.mean(axis=0),x2.mean(axis=0))

from sklearn.model_selection import train_test_split
X=np.concatenate((x1,x2),axis=0)
Y=np.concatenate((np.zeros(len(x1)),np.ones(len(x1))),axis=0 )
X,X_test,Y,Y_test=train_test_split(X,Y)
print (X.shape,Y.shape)

#data = np.loadtxt("variational_classifier/data/parity_train.txt", dtype=int)

#X = np.array(data[:, :-1])
#Y = np.array(data[:, -1])
Y = Y * 2 - 1  # shift label from {0, 1} to {-1, 1}
c=0
for x,y in zip(X, Y):
    print(f"x = {x}, y = {y}")
    c+=1
    if c==10: break

"""We initialize the variables randomly (but fix a seed for
reproducibility). Remember that one of the variables is used as a bias,
while the rest is fed into the gates of the variational circuit.

"""

np.random.seed(0)
num_qubits = 4
num_layers = 2
weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

print("Weights:", weights_init)
print("Bias: ", bias_init)

"""Next we create an optimizer instance and choose a batch size...

"""

opt = NesterovMomentumOptimizer(0.01)
batch_size = 5

"""...and run the optimizer to train our model. We track the accuracy - the
share of correctly classified data samples. For this we compute the
outputs of the variational classifier and turn them into predictions in
$\{-1,1\}$ by taking the sign of the output.

"""

weights = weights_init
bias = bias_init
for it in range(10):

    # Update the weights by one optimizer step, using only a limited batch of data
    batch_index = np.random.randint(0, len(X), (batch_size,))
    X_batch = X[batch_index]
    Y_batch = Y[batch_index]
    weights, bias = opt.step(cost, weights, bias, X=X_batch, Y=Y_batch)

    # Compute accuracy
    predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X]

    current_cost = cost(weights, bias, X, Y)
    acc = accuracy(Y, predictions)

    print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")

"""As we can see, the variational classifier learned to classify all bit
strings from the training set correctly.

But unlike optimization, in machine learning the goal is to generalize
from limited data to *unseen* examples. Even if the variational quantum
circuit was perfectly optimized with respect to the cost, it might not
generalize, a phenomenon known as *overfitting*. The art of (quantum)
machine learning is to create models and learning procedures that tend
to find \"good\" minima, or those that lead to models which generalize
well.

With this in mind, let\'s look at a test set of examples we have not
used during training:

"""

Y_test = Y_test * 2 - 1  # shift label from {0, 1} to {-1, 1}

predictions_test = [np.sign(variational_classifier(weights, bias, x)) for x in X_test]
c=0
for x,y,p in zip(X_test, Y_test, predictions_test):
    c+=1
    if c==10: break
    print(f"x = {x}, y = {y}, pred={p}")

acc_test = accuracy(Y_test, predictions_test)
print("Accuracy on unseen data:", acc_test)

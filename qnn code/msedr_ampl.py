seed = 1

import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optax
from functools import partial

from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

np.random.seed(seed)

raw = pd.read_csv("diabetes.csv")
test = raw[raw['Insulin']!=0]
test2 = test[test['Glucose']!=0]
data = test2[test['BMI']!=0]

X=data.iloc[:,:-1].to_numpy()
Y=data.iloc[:,-1].to_numpy()

PredictorScaler=StandardScaler()
PredictorScalerFit=PredictorScaler.fit(X)
X=PredictorScalerFit.transform(X)
Y = Y * 2 - np.ones(len(Y))  # shift label from {0, 1} to {-1, 1}

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.7)
X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
Y_train = jnp.array(Y_train)
Y_test = jnp.array(Y_test)

def data_encoding(features):
    qml.MottonenStatePreparation(state_vector=features, wires=range(3)) 

def layer(W):
    qml.CZ(wires=[0,1])
    qml.CZ(wires=[0,2])

    qml.broadcast(qml.RY, wires=range(3), parameters=W[0:3], pattern='single')
    
    qml.CZ(wires=[1,2])
    qml.CZ(wires=[2,0])
    
    qml.broadcast(qml.RY, wires=range(3), parameters=W[3:6], pattern='single')

dev = qml.device('default.qubit.jax', wires=3)

@qml.qnode(dev, interface="jax")
def circuit(weights, x):
    data_encoding(x)
    layer(weights[0:6])

    for i in range(1,layers):
        data_encoding(x)
        layer(weights[6*i: 6*i+6])
    return qml.expval(qml.PauliZ(0)) 

def variational_classifier(weights, x):
    return circuit(weights[:-1],x) + weights[-1]
    
circuit_batched = jax.vmap(variational_classifier, (0, None))
circuit_jit = jax.jit(variational_classifier)

@jax.jit
def square_loss(labels, predictions): 
    yp = jnp.array(predictions)
    y = jnp.array(labels)
    cost = jnp.mean((yp - y) ** 2)
    return cost

@jax.jit
def accuracy(labels, predictions):
    return sum(jnp.isclose(jnp.array(labels), jnp.array(predictions)))/len(labels)

@jax.jit
def cost(weights, X, Y):
    predictions = [circuit_jit(weights, x) for x in X]
    return square_loss(Y, predictions)

@jax.jit
def optimizer_update(opt_state, weights, x, y):
    loss, grads = jax.value_and_grad(lambda theta: cost(theta, x, y))(weights)    
    updates, opt_state = optimizer.update(grads, opt_state)
    weights = optax.apply_updates(weights, updates)
    return weights, opt_state, loss

layers = 6
weights_init = 0.01 * np.random.randn(6*layers + 1, requires_grad=True)

epochs = 500
batch_size = 50
stocastic_step = int(X_train.shape[0]/batch_size)
lr = 0.001

optimizer = optax.adam(learning_rate=lr)

weights = jnp.copy(weights_init)

key = jax.random.PRNGKey(0)
key = jax.random.split(key)[0]

opt_state = optimizer.init(weights)

train_costs = []
train_accs = []
test_costs = []
test_accs = []

for epoch in range(epochs):
    for st in range(stocastic_step):

        idxs = jax.random.choice(key, jnp.array(list(range(X_train.shape[0]))), shape=(batch_size,))
        key = jax.random.split(key)[0]

        weights, opt_state, cost = optimizer_update(opt_state, weights, X_train[idxs], Y_train[idxs])

    # Compute accuracy_train
    predictions_train = [circuit_jit(weights, x) for x in X_train]
    cost_train = square_loss(predictions_train, Y_train)
    train_costs.append(cost_train)
    acc_train = accuracy(Y_train, np.sign(predictions_train))
    train_accs.append(acc_train)
    
    # Compute accuracy_val
    predictions_val = [circuit_jit(weights, x) for x in X_test]
    cost_val = square_loss(predictions_val, Y_test)
    test_costs.append(cost_val)
    acc_val = accuracy(Y_test, np.sign(predictions_val))
    test_accs.append(acc_val)

fname_train_acc = f'Results/MSE_DataReuploading/train_acc_{layers}layers_lerningrate{lr}_batchsize{batch_size}_stocstep{stocastic_step}_{epochs}epochs_{seed}seed_MSE_DataReuploading.txt'

with open(fname_train_acc,'w') as f:
    for i in train_accs:
        f.write(str(i))
        f.write('\n')

fname_test_acc = f'Results/MSE_DataReuploading/test_acc_{layers}layers_lerningrate{lr}_batchsize{batch_size}_stocstep{stocastic_step}_{epochs}epochs_{seed}seed_MSE_DataReuploading.txt'

with open(fname_test_acc,'w') as f:
    for i in test_accs:
        f.write(str(i))
        f.write('\n')

fname_train_cost = f'Results/MSE_DataReuploading/train_cost_{layers}layers_lerningrate{lr}_batchsize{batch_size}_stocstep{stocastic_step}_{epochs}epochs_{seed}seed_MSE_DataReuploading.txt'

with open(fname_train_cost,'w') as f:
    for i in train_costs:
        f.write(str(i))
        f.write('\n')

fname_test_cost = f'Results/MSE_DataReuploading/test_cost_{layers}layers_lerningrate{lr}_batchsize{batch_size}_stocstep{stocastic_step}_{epochs}epochs_{seed}seed_MSE_DataReuploading.txt'

with open(fname_test_cost,'w') as f:
    for i in test_costs:
        f.write(str(i))
        f.write('\n')

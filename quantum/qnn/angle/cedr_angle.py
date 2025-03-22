import sys

layers =  int(sys.argv[1])
seed = int(sys.argv[2])

import pennylane as qml

from pennylane import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optax
from functools import partial
from sklearn.compose import ColumnTransformer

import jax
import jax.numpy as jnp

np.random.seed(seed)

data = pd.read_csv('heart.csv')

data['Sex'].replace(['M', 'F'],
                        [0, 1], inplace=True)

data['ExerciseAngina'].replace(['N', 'Y'],
                        [0, 1], inplace=True)

data['ChestPainType'].replace(['ATA', 'NAP', 'ASY', 'TA'], [0,1,2,3], inplace=True)

data['RestingECG'].replace(['Normal', 'ST', 'LVH'], [0,1,2], inplace=True)

data['ST_Slope'].replace(['Up', 'Flat', 'Down'], [0,1,2], inplace=True)

Y=data['HeartDisease'].to_numpy()
X=data.drop(['HeartDisease'], axis = 1)
X = np.pi*X / abs(X).max(axis=0)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.7, random_state=2)
X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
Y_train = jnp.array(Y_train)
Y_test = jnp.array(Y_test)

def data_encoding(features):
    qml.AngleEmbedding(features=features, wires=range(int(X_train.shape[1])), rotation='Y')

def layer(W):
    n = int(X_train.shape[1])
    qubit_wires = range(n)

    qml.broadcast(qml.CZ,wires=qubit_wires, pattern='double')
    qml.broadcast(qml.RY, wires=qubit_wires, parameters=W[0:n], pattern='single')
    qml.broadcast(qml.CZ,wires=qubit_wires[1:], pattern='double')
    qml.broadcast(qml.RY, wires=qubit_wires[1:-1], parameters=W[n:2*(n-1)], pattern='single')

dev = qml.device('default.qubit.jax', wires=range(int(X_train.shape[1])))

@qml.qnode(dev, interface="jax")
def circuit(weights, x):
    n = int(X_train.shape[1])
    val = 2*(n - 1)

    data_encoding(x)
    layer(weights[0:val])

    for l in range(1, layers):
        data_encoding(x)
        layer(weights[val*l:val*l+val])

    return [qml.expval(qml.PauliZ(i)) for i in range(n)]

def variational_classifier(weights, x):
    res = jnp.sum(jnp.array(circuit(weights[:-1],x)))
    return res + weights[-1]

circuit_batched = jax.vmap(variational_classifier, (0, None))
circuit_jit = jax.jit(variational_classifier)

@jax.jit
def cross_entropy_loss(labels, predictions): 
    yp = jnp.array(predictions)
    y = jnp.array(labels)

    cost = -1*jnp.mean(y*jnp.log(yp) + (1-y)*jnp.log(1-yp))
    return cost

@jax.jit
def accuracy(labels, predictions):
    return sum(jnp.isclose(jnp.array(labels), jnp.array(predictions)))/len(labels)

@jax.jit
def cost(weights, X, Y):
    predictions = jnp.array([circuit_jit(weights, x) for x in X])
    predictions = jax.nn.sigmoid(predictions)
    return cross_entropy_loss(Y, predictions)

@jax.jit
def optimizer_update(opt_state, weights, x, y):
    loss, grads = jax.value_and_grad(lambda theta: cost(theta, x, y))(weights)    
    updates, opt_state = optimizer.update(grads, opt_state)
    weights = optax.apply_updates(weights, updates)
    return weights, opt_state, loss

n = int(X_train.shape[1])
val = 2*(n - 1)

weights_init = 0.01 * np.random.randn(val*layers + 1, requires_grad=True)

batch_size = 64
stocastic_step = int(X_train.shape[0]/batch_size)

lr = 0.01
optimizer = optax.adam(learning_rate=lr)

weights = jnp.copy(weights_init)

key = jax.random.PRNGKey(0)
key = jax.random.split(key)[0]

opt_state = optimizer.init(weights)

train_costs = []
train_accs = []
test_costs = []
test_accs = []

condition = True

patience_lr = 50
count_lr = 0

patience_stopping = 100
count_stopping = 0

min_cost = 10
epoch = 0
# for epoch in range(epochs):
while condition:
    for st in range(stocastic_step):

        idxs = jax.random.choice(key, jnp.array(list(range(X_train.shape[0]))), shape=(batch_size,))
        key = jax.random.split(key)[0]

        weights, opt_state, cost = optimizer_update(opt_state, weights, X_train[idxs], Y_train[idxs])

    # Compute accuracy_train
    predictions_train = [circuit_jit(weights, x) for x in X_train]
    cost_train = cross_entropy_loss(Y_train, jax.nn.sigmoid(jnp.array(predictions_train)))
    train_costs.append(cost_train)
    acc_train = accuracy(Y_train, (np.sign(predictions_train) + 1)/2)
    train_accs.append(acc_train)
    
    # Compute accuracy_val
    predictions_val = [circuit_jit(weights, x) for x in X_test]
    cost_val = cross_entropy_loss(Y_test, jax.nn.sigmoid(jnp.array(predictions_val)))
    test_costs.append(cost_val)

    acc_val = accuracy(Y_test, (np.sign(predictions_val) + 1)/2)
    test_accs.append(acc_val)

    if cost_val < min_cost:
        min_cost = cost_val
        count_lr = 0
        count_stopping = 0
    else:
        count_lr += 1
        count_stopping += 1
    
    if count_lr == patience_lr:
        lr = lr*0.5
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init(weights)

    if count_stopping == patience_stopping:
        condition = False

    epoch += 1
# print(weights[-1])

fname_train_acc = f'Results/Cross_Entropy_DataReuploading/train_acc_{layers}layers_batchsize{batch_size}_stocstep{stocastic_step}_{seed}seed_CE.txt'

with open(fname_train_acc,'w') as f:
    for i in train_accs:
        f.write(str(i))
        f.write('\n')

fname_test_acc = f'Results/Cross_Entropy_DataReuploading/test_acc_{layers}layers_batchsize{batch_size}_stocstep{stocastic_step}_{seed}seed_CE.txt'

with open(fname_test_acc,'w') as f:
    for i in test_accs:
        f.write(str(i))
        f.write('\n')

fname_train_cost = f'Results/Cross_Entropy_DataReuploading/train_cost_{layers}layers_batchsize{batch_size}_stocstep{stocastic_step}_{seed}seed_CE.txt'

with open(fname_train_cost,'w') as f:
    for i in train_costs:
        f.write(str(i))
        f.write('\n')

fname_test_cost = f'Results/Cross_Entropy_DataReuploading/test_cost_{layers}layers_batchsize{batch_size}_stocstep{stocastic_step}_{seed}seed_CE.txt'

with open(fname_test_cost,'w') as f:
    for i in test_costs:
        f.write(str(i))
        f.write('\n')

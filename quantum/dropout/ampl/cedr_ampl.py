import sys

layers = int(sys.argv[1])
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

data = pd.read_csv("heart_final.csv")

Y=data['HeartDisease'].to_numpy()
X=data.drop(['Unnamed: 0','HeartDisease'], axis = 1)

ct = ColumnTransformer([('stand_scal', StandardScaler(), ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'] )], remainder='passthrough')
X = ct.fit_transform(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.7, random_state=2)
X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
Y_train = jnp.array(Y_train)
Y_test = jnp.array(Y_test)

def true_cond(angle):
    return 0.0

def false_cond(angle):
    return angle

def data_encoding(features):
    ampl_vec = features[0:8] / jnp.linalg.norm(features[0:8])
    basis_vec =features[8:] 

    qml.MottonenStatePreparation(ampl_vec, wires=range(3))
    for idx, el in enumerate(basis_vec):
        qml.RY(jnp.pi*el, wires=3+idx)
        

def var_ansatz(
    weights, rotations=[qml.RY], entangler=qml.CNOT, keep_rotation=None
):

    """
    Single layer of the variational ansatz for our QNN.
    We have a single qubit rotation per each qubit (wire) followed by
    a linear chain of entangling gates (entangler). This structure is repeated per each rotation in `rotations`
    (defining `inner_layers`).
    The single qubit rotations are applied depending on the values stored in `keep_rotation`:
    if the value is negative the rotation is dropped (rotation dropout), otherwise it is applied.

    Params:
    - theta: variational angles that will undergo optimization
    - wires: list of qubits (wires)
    - rotations: list of rotation kind per each `inner_layer`
    - entangler: entangling gate
    - keep_rotation: list of lists. There is one list per each `inner_layer`.
                    In each list there are indexes of the rotations that we want to apply.
                    Some of these values may be substituted by -1 value
                    which means that the rotation gate wont be applied (dropout).
    """

    n = n_qubits
    wires = range(n)

    counter = 0
    rot = rotations[0]
    # print(keep_rotation)
    for qb, keep_or_drop in enumerate(keep_rotation[0]):
        angle = weights[counter] 
        angle_drop = jax.lax.cond(keep_or_drop < 0, true_cond, false_cond, angle)
        rot(angle_drop, wires=wires[qb])
        counter += 1
    qml.broadcast(qml.CZ,wires=wires, pattern='double')

    for qb, keep_or_drop in enumerate(keep_rotation[1]):
        angle = weights[counter] 
        angle_drop = jax.lax.cond(keep_or_drop < 0, true_cond, false_cond, angle)
        rot(angle_drop, wires=wires[qb+1])
        counter += 1
    qml.broadcast(qml.CZ,wires=wires[1:], pattern='double')

def create_circuit(n_qubits):
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device)
    def circuit(x, theta, keep_rot):
        n = n_qubits
        params_per_layer = 2*(n - 1)
        for i in range(layers):
            data_encoding(x)

            keep_rotation = keep_rot[i]

            var_ansatz(
                theta[i * params_per_layer : (i + 1) * params_per_layer],
                entangler=qml.CNOT,
                keep_rotation=keep_rotation,
            )

        return [qml.expval(qml.PauliZ(i)) for i in range(n)]
    return circuit

def variational_classifier(weights, x, keep_rot):
    n_qubits = int(X_train.shape[1])
    circ = create_circuit(n_qubits)
    res = jnp.sum(jnp.array(circ(x, weights[:-1], keep_rot)))
    return res + weights[-1]

circuit_batched = jax.vmap(variational_classifier, (0, None, None))
circuit_jit = jax.jit(variational_classifier)

def make_dropout(key):
    drop_layers = []

    for lay in range(layers):
        # each layer has prob p_L=layer_drop_rate of being dropped
        # according to that for every layer we sample
        # if we have to appy dropout in it or not
        out = jax.random.choice(
            key, jnp.array(range(2)), p=jnp.array([1 - layer_drop_rate, layer_drop_rate])
        )
        key = jax.random.split(key)[0]  # update the random key

        if out == 1:  # if it has to be dropped
            drop_layers.append(lay)
    keep_rot = []
    # we make list of indexes corresponding to the rotations gates
    # that are kept in the computation during a single train step
    for i in range(layers):
        # each list is divded in layers and then in "inner layers"
        # this is strictly related to the QNN architecture that we use
        keep_rot_layer = [list(range((n_qubits - 2*(j-1)))) for j in range(1, inner_layers + 1)]
        # print(keep_rot_layer)
        if i in drop_layers:  # if dropout has to be applied in this layer
            keep_rot_layer = []  # list of indexes for a single layer
            inner_keep_r = []  # list of indexes for a single inner layer
            for param in range(params_per_layer):
                # each rotation within the layer has prob p=rot_drop_rate of being dropped
                # according to that for every parameter (rotation) we sample
                # if we have to drop it or not
                out = jax.random.choice(
                    key, jnp.array(range(2)), p=jnp.array([1 - rot_drop_rate, rot_drop_rate])
                )
                key = jax.random.split(key)[0]  # update the random key
                # print(out)
                if out == 0:  # if we have to keep it
                    inner_keep_r.append(param % n_qubits)  # % is required because we work
                    # inner layer by inner layer
                else:  # if the rotation has to be dropped
                    inner_keep_r.append(-1)  # we assign the value -1

                # if param % n_qubits == n_qubits - 1:  # if it's the last qubit of the register
                if len(inner_keep_r) == n_qubits:
                    # append the inner layer list
                    keep_rot_layer.append(inner_keep_r)
                    # and reset it
                    inner_keep_r = []
                elif len(inner_keep_r) == n_qubits - 2 and len(keep_rot_layer)==1:
                    keep_rot_layer.append(inner_keep_r)
                    # and reset it
                    inner_keep_r = []
        keep_rot.append(keep_rot_layer)
    # return jnp.array(keep_rot)
    return keep_rot

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
def cost(weights, X, Y): # this has to change to include modifications    
    predictions = jnp.array([circuit_jit(weights, x, keep_rot) for x in X])
    predictions = jax.nn.sigmoid(predictions)
    return cross_entropy_loss(Y, predictions)

@jax.jit
def optimizer_update(opt_state, weights, x, y):
    loss, grads = jax.value_and_grad(lambda theta: cost(theta, x, y))(weights)   
    updates, opt_state = optimizer.update(grads, opt_state)
    weights = optax.apply_updates(weights, updates)
    return weights, opt_state, loss

n_qubits = int(X_train.shape[1])
inner_layers = 2
params_per_layer = 2*(n_qubits - 1)

rot_drop_rate = 0.2
layer_drop_rate = 0.3

weights_init = 0.01 * np.random.randn(params_per_layer*layers + 1, requires_grad=True)

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

        keep_rot = make_dropout(key) 
        key = jax.random.split(key)[0]

        weights, opt_state, cost = optimizer_update(opt_state, weights, X_train[idxs], Y_train[idxs])

    keep_all_rot = [[list(range((n_qubits - 2*(j-1)))) for j in range(1, inner_layers + 1)],[list(range((n_qubits - 2*(j-1)))) for j in range(1, inner_layers + 1)]] 
    
    # Compute accuracy_train
    predictions_train = [circuit_jit(weights, x, keep_rot) for x in X_train]
    cost_train = cross_entropy_loss(Y_train, jax.nn.sigmoid(jnp.array(predictions_train)))
    train_costs.append(cost_train)
    acc_train = accuracy(Y_train, (np.sign(predictions_train) + 1)/2)
    train_accs.append(acc_train)
    
    # Compute accuracy_val
    predictions_val = [circuit_jit(weights, x, keep_rot) for x in X_test]
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

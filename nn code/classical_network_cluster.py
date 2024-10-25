import sys

layers = int(sys.argv[1])
seed = int(sys.argv[2])
n_nodes = int(sys.argv[3])

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

raw = pd.read_csv("diabetes.csv")
test = raw[raw['Insulin']!=0]
test2 = test[test['Glucose']!=0]
data = test2[test['BMI']!=0]

#dataset has to be cleaned
X=data.iloc[:,:-1].to_numpy()
Y=data.iloc[:,-1].to_numpy()
Y = np.array([Y[i] for i in range(len(Y))])

PredictorScaler=StandardScaler()
PredictorScalerFit=PredictorScaler.fit(X)
X=PredictorScalerFit.transform(X)

np.random.seed(seed)

train_size = 0.7
X_train, X_test,Y_train, Y_test = train_test_split(X,Y,train_size = train_size, random_state=11) 

X_train = tf.convert_to_tensor(X_train, dtype=tf.float64)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float64)
Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float64)
Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float64)

model = Sequential()
if layers == 0:
    model.add(Dense(1, input_dim=8, activation='sigmoid'))
else:
    model.add(Dense(n_nodes, input_dim=8, activation='sigmoid'))
    for i in range(layers):
        model.add(Dense(n_nodes, activation='sigmoid'))
    model.add(Dense(1, activation='linear'))

learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=1000, batch_size=50, validation_data=(X_test, Y_test), verbose=False)

fname_train_losses = f'Results/Classical_network_{train_size}trainsize_trainloss_{seed}seed_{layers}layers.txt'
with open(fname_train_losses,'w') as f:
    for l in history.history['loss']:
        f.write(str(l))
        f.write('\n')

fname_test_losses = f'Results/Classical_network_{train_size}trainsize_testloss_{seed}seed_{layers}layers.txt'
with open(fname_test_losses,'w') as f:
    for l in history.history['val_loss']:
        f.write(str(l))
        f.write('\n')

fname_train_accs = f'Results/Classical_network_{train_size}trainsize_trainaccs_{seed}seed_{layers}layers.txt'
with open(fname_train_accs,'w') as f:
    for l in history.history['accuracy']:
        f.write(str(l))
        f.write('\n')

fname_test_accs = f'Results/Classical_network_{train_size}trainsize_testaccs_{seed}seed_{layers}layers.txt'
with open(fname_test_accs,'w') as f:
    for l in history.history['val_accuracy']:
        f.write(str(l))
        f.write('\n')

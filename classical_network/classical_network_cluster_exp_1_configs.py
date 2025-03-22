import sys
import ast


node_list = ast.literal_eval(sys.argv[1])
seed = int(sys.argv[2])

# from typing import List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import tensorflow as tf
# from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
# from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

data = pd.read_csv("heart_final.csv")

Y=data['HeartDisease'].to_numpy()
X=data.drop(['HeartDisease'], axis = 1)

ct = ColumnTransformer([('stand_scal', StandardScaler(), ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'] )], remainder='passthrough')
X = ct.fit_transform(X)

train_size = 0.7
X_train, X_test,Y_train, Y_test = train_test_split(X, Y, train_size = train_size, random_state=2) 

X_train = tf.convert_to_tensor(X_train, dtype=tf.float64)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float64)
Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float64)
Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float64)

np.random.seed(seed)
tf.random.set_seed(seed)

initializer_zeros = tf.keras.initializers.Zeros()
initializer_HEnormal = tf.keras.initializers.HeNormal(seed=seed)
initializer_HEuniform = tf.keras.initializers.HeUniform(seed=seed)
initializer_xavier = tf.keras.initializers.GlorotNormal(seed=seed)

model = Sequential()
if len(node_list) == 0:
    model.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid', bias_initializer='zeros', kernel_initializer = initializer_HEuniform))
else:
    model.add(Dense(node_list[0], input_dim=X_train.shape[1], activation= tf.keras.layers.LeakyReLU(alpha=0.1), bias_initializer='zeros', kernel_initializer = initializer_HEnormal))
    for i in range(1, len(node_list)):
        model.add(Dense(node_list[i], activation=tf.keras.layers.LeakyReLU(alpha=0.1), bias_initializer='zeros', kernel_initializer = initializer_HEnormal))
	
    model.add(Dense(1, activation='sigmoid', bias_initializer='zeros', kernel_initializer = initializer_xavier))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=500,        
    restore_best_weights=True,
    start_from_epoch = 100
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, min_lr=1e-7)

history = model.fit(X_train, Y_train, epochs=10000, batch_size=64, validation_data=(X_test, Y_test), verbose=False, callbacks=[lr_scheduler, early_stopping])

layers = len(node_list)
n_nodes = ''
for i in node_list:
    n_nodes += str(i)

fname_train_losses = f'Results/Res_deep_2/Classical_network_{train_size}trainsize_trainloss_{seed}seed_{layers}layers_{n_nodes}nodes_Adam.txt'
with open(fname_train_losses,'w') as f:
    for l in history.history['loss']:
        f.write(str(l))
        f.write('\n')

fname_test_losses = f'Results/Res_deep_2/Classical_network_{train_size}trainsize_testloss_{seed}seed_{layers}layers_{n_nodes}nodes_Adam.txt'
with open(fname_test_losses,'w') as f:
    for l in history.history['val_loss']:
        f.write(str(l))
        f.write('\n')

fname_train_accs = f'Results/Res_deep_2/Classical_network_{train_size}trainsize_trainaccs_{seed}seed_{layers}layers_{n_nodes}nodes_Adam.txt'
with open(fname_train_accs,'w') as f:
    for l in history.history['accuracy']:
        f.write(str(l))
        f.write('\n')

fname_test_accs = f'Results/Res_deep_2/Classical_network_{train_size}trainsize_testaccs_{seed}seed_{layers}layers_{n_nodes}nodes_Adam.txt'
with open(fname_test_accs,'w') as f:
    for l in history.history['val_accuracy']:
        f.write(str(l))
        f.write('\n')

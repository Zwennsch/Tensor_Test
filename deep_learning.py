import numpy as np
import tensorflow as tf
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy


# 1. preparing/generating the data
train_labels = []
train_samples =[]

for i in range(50):
    # 5% of individuals with side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # 5% of individuals without side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # 95% of individuals with side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # 95% of individuals without side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

# 2. Processing the data:
train_labels = np.array(train_labels)
train_samples= np.array(train_samples)

# get rid of the order while maintaining the alignment between both datasets:
train_labels, train_samples = shuffle(train_labels, train_samples)

# transform the ages from 13..100 -> 0..1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

# for i in scaled_train_samples:
#     print(i)



# 3. Building the model (a Sequential model)
# A Sequential model is a linear stack of layers 
# The first layer, the input layer, is omitted, since the input data is what creates the input layer itself
model = Sequential([
    Dense(units=16, activation='relu'), # 16 units/neurons is arbitrary. This is the first hidden layer
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')  # this is the output layer. units=2 corresponds to the two output classes either did experience side-effects or did not experience side effects. the 'softmax' function gives probabilities for each output class
])



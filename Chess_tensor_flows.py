! rm games*
! wget https://raw.githubusercontent.com/dstjohniii/Chess-AI-Analysis/main/games.csv
import numpy as np
dataset = np.genfromtxt('games.csv', delimiter=',', skip_header=True, usecols=(5, 6, 9))
white_rating_max = dataset[:, 0].max()
black_rating_max = dataset[:, 1].max()

#normalize by dividing by max
dataset[:, 0] = dataset[:, 0] / white_rating_max
dataset[:, 1] = dataset[:, 1] / black_rating_max

# Shuffle the datasets
import random
np.random.shuffle(dataset)

X = dataset[:, :-1]
Y = dataset [:, -1]

# Split into training and validation, 30% validation set and 70% training 
index_30percent = int(0.3 * len(dataset[:, 0]))
print(index_30percent)

XVALID = X[:index_30percent]
YVALID = Y[:index_30percent]
XTRAIN = X[index_30percent:]
YTRAIN = Y[index_30percent:]
print(XVALID.shape, YVALID.shape, XTRAIN.shape, YTRAIN.shape)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# File name must be in quotes
callback_a = ModelCheckpoint(filepath = "Chess.hdf5", monitor='val_loss', save_best_only = True, save_weights_only = True, verbose = 0)
# The patience value can be 10, 20, 100, etc. depending on when your model starts to overfit
callback_b = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=0)

# Two layer 4,1 neuron
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(4, input_dim = len(XTRAIN[0, :]), activation='relu'))
model.add(Dense(1, activation='sigmoid')) ##sigmoid has to be used for classification.

model.compile(loss='binary_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])

# Do the training (specify the validation set as well)
history = model.fit(XTRAIN, YTRAIN, validation_data = (XVALID, YVALID), epochs = 100, batch_size = 16, verbose = 0, 
                    callbacks = [callback_a, callback_b])
# Check what's in the history
print(history.params)

# load checkpoint-ed model
model.load_weights("Chess.hdf5")

# Plot the Accuracy learning curves
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy']) # replace with accuracy/MAE
plt.plot(history.history['val_accuracy']) # replace with val_accuracy, etc.
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['training data', 'validation data'], loc='lower right')
plt.show()
# Plot the Loss Learning curve
plt.plot(history.history['loss']) # replace with accuracy/MAE
plt.plot(history.history['val_loss']) # replace with val_accuracy, etc.
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['training data', 'validation data'], loc='upper right')
plt.show()

# Evaluations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
prediction = model.predict(X)
accuracy = accuracy_score(Y, prediction.round())
precision = precision_score(Y, prediction.round())
recall = recall_score(Y, prediction.round())
f1score = f1_score(Y, prediction.round())
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Precision: %.2f%%" % (precision * 100.0))
print("Recall: %.2f%%" % (recall * 100.0))
print("F1-score: %.2f" % (f1score))
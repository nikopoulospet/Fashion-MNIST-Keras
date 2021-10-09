import keras.utils.np_utils
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.regularizers import l1_l2, l1, l2

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


debug = False

# ++++++++++ data preprocessing ++++++++++++++++++++++++++++++++++
# load data from .npy files (static paths)
images = np.load("images.npy")  # np shape: (6500, 784) -> 784 = 28*28
labels = np.load("labels.npy")  # np shape: (6500, )
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# set image data to 0-1 scale
images = images / 255.0

# one hot encoding for labels
labels = keras.utils.np_utils.to_categorical(labels, num_classes=10)

# split data into Train, Val, and test sets. (.6,.15,.25)
shuffle = np.random.permutation(labels.shape[0])
images = images[shuffle]
labels = labels[shuffle]

train = int(labels.shape[0] * .6)
val = int(labels.shape[0] * .15) + train

x_train = images[:train]
y_train = labels[:train]

x_val = images[train:val]
y_val = labels[train:val]

x_test = images[val:]
y_test = labels[val:]

if debug:
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(class_names[y_train[i]])
    plt.show()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Model Template
model = Sequential()  # declare model
model.add(Dense(28*28, input_shape=(28*28, ),
          kernel_initializer='he_normal'))  # first layer
# model.add(Activation('relu'))
model.add(Dense(28*28, kernel_initializer='he_normal'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#
# Fill in Model Here
#
#
# Initialize weights randomly for every layer, try different initialization schemes.
# Experiment with using ReLu Activation Units, as well as SeLu and Tanh.
# Experiment with number of layers and number of neurons in each layer, including the first layer.

model.add(Dense(10, kernel_initializer='he_normal'))  # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=10,
                    batch_size=256)

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
indices = [i for i, v in enumerate(pred) if pred[i] != y_test[i]]
subset_of_wrongly_predicted = [x_test[i] for i in indices]

# get three images
image_one = subset_of_wrongly_predicted[0]
image_two = subset_of_wrongly_predicted[1]
image_three = subset_of_wrongly_predicted[2]
for i in range(3):
    im = subset_of_wrongly_predicted[i]
    plt.imshow(np.reshape(im, [28, 28]), cmap='gray')
    plt.show()

# epochs v accuracy, val_accuracy
fig, ax = plt.subplots()
ax.plot(np.arange(0, int(history.params['epochs'])),
        history.history['accuracy'], label='accuracy')
ax.plot(np.arange(0, int(history.params['epochs'])),
        history.history['val_accuracy'], label='val_accuracy')
ax.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

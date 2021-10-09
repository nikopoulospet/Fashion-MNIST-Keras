import keras.utils.np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Dropout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


debug = False

# ++++++++++ data preprocessing ++++++++++++++++++++++++++++++++++
# load data from .npy files (static paths)
images = np.load("images.npy")  # np shape: (6500, 78q4) -> 784 = 28*28
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

# network parameters
dropOut = 0.15

model = Sequential()  # declare model
model.add(Dense(784, input_dim = 784))  # first layer
model.add(Activation('relu'))
model.add(Dropout(dropOut))

model.add(Dense(784))
model.add(Activation('relu'))
model.add(Dropout(dropOut))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(dropOut))

model.add(Dense(10))
model.add(Activation('softmax'))

# Initialize weights randomly for every layer, try different initialization schemes.
# Experiment with using ReLu Activation Units, as well as SeLu and Tanh.
# Experiment with number of layers and number of neurons in each layer, including the first layer.

# Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=15,
                    batch_size=512)


# Report Results

print(history.history)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Confusion Matrix
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test, y_pred)
print(cm)

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
indices = [i for i,v in enumerate(pred) if pred[i]!=y_test[i]]
subset_of_wrongly_predicted = [x_test[i] for i in indices]

# get three images
image_one = subset_of_wrongly_predicted[0]
image_two = subset_of_wrongly_predicted[1]
image_three = subset_of_wrongly_predicted[2]
for i in range(3):
    im = subset_of_wrongly_predicted[i]
    plt.imshow(np.reshape(im,[28,28]), cmap='gray')
    plt.show()
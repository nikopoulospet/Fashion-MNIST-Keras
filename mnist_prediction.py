import keras.utils.np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import numpy as np

debug = False

# ++++++++++ data preprocessing ++++++++++++++++++++++++++++++++++
#load data from .npy files (static paths)
images = np.load("images.npy") # np shape: (6500, 784) -> 784 = 28*28
labels = np.load("labels.npy") # np shape: (6500, )
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#set image data to 0-1 scale
images = images / 255.0

#one hot encoding for labels
labels = keras.utils.np_utils.to_categorical(labels, num_classes=10)

#split data into Train, Val, and test sets. (.6,.15,.25)
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
        plt.imshow(x_train[i].reshape(28,28), cmap=plt.cm.binary)
        plt.xlabel(class_names[y_train[i]])
    plt.show()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Model Template

model = Sequential() # declare model
model.add(Dense(28*28, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))
model.add(Dense(28*28, kernel_initializer='he_normal'))
model.add(Activation('relu'))
#
# Fill in Model Here
#
#
model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train, 
                    validation_data = (x_val, y_val), 
                    epochs=50,
                    batch_size=512)


# Report Results

print(history.history)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
# figure out confusion matrix
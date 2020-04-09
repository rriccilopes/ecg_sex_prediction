# -*- coding: utf-8 -*-
'''
Created on Thu Apr  9 11:31:29 2020

@author: rriccilopes
'''

# Load data
import os
from matplotlib import pyplot as plt

from keras import layers, models
from keras.optimizers import Adam

from data_generator import DataGenerator

# Base directory with data
base_dir = 'C:/Data/Age_and_sex/'
target_names = ['male', 'female']

# Samples split from split_data.py
train_path = base_dir + 'train/'
test_path = base_dir + 'test/'
val_path = base_dir + 'validation/'

# Assert if can load data
for data_path in [train_path, test_path, val_path]:
    for label in target_names:
        files = os.listdir(data_path + label)
        assert len(files) > 0, 'Not able to load files: ' + data_path + label
del files, data_path, label

# Create dicts
partition = {'train':[], 'validation':[], 'test':[]}
labels = {}

for data_path, part in zip([train_path, test_path, val_path],
                           ['train', 'test', 'validation']):
    for label in target_names:
        for file_path in os.listdir(data_path + label):
            partition[part].append(data_path + label + '/' + file_path)
            labels.update({data_path + label + '/' + file_path: label})

del data_path, part, label, file_path

# %% Create data loader
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random
#from keras.applications.resnet50 import ResNet50
#model = ResNet50()
#model.summary()

params = {'dim':(8, 2500), 'n_channels':1, 'n_classes':2,
          'batch_size':64, 'shuffle':True}

#train_gen = DataGenerator(random.sample(partition['train'], 500), labels, target_names=target_names, **params)
#val_gen = DataGenerator(random.sample(partition['validation'], 200), labels, target_names=target_names,**params)
#test_gen = DataGenerator(random.sample(partition['test'], 200), labels, target_names=target_names, **params)

train_gen = DataGenerator(partition['train'], labels, target_names=target_names, **params)
val_gen = DataGenerator(partition['validation'], labels, target_names=target_names,**params)
test_gen = DataGenerator(partition['test'], labels, target_names=target_names, **params)


# Design model
# Adapted from https://www.ahajournals.org/doi/full/10.1161/CIRCEP.119.007284
model = models.Sequential()
n, k, mp = 16, 7, 2
model.add(layers.Conv2D(n, (1, k), activation='relu',
                        input_shape=(8, 2500, 1)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D((1, mp)))

# Temporal analysis
for n, k, mp in zip([16, 32, 32, 64, 64, 64, 64],
                    [5, 5, 5, 5, 3, 3, 3, 3],
                    [4, 2, 4, 2, 2, 2]):
    model.add(layers.Conv2D(n, (1, k), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((1, mp)))

# Spatial analysis
model.add(layers.Conv2D(128, (8, 1), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D((1, 2)))

# FC Layers
for n in [128, 64]:
    model.add(layers.Dense(n, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))

# Output
model.add(layers.Flatten())
model.add(layers.Dense(2, activation="softmax"))


optimizer = Adam(lr=0.0003)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=5)

model_fp = "models/saved-model-{epoch:02d}-{val_loss:.2f}.hdf5"
cp = ModelCheckpoint(model_fp, monitor='val_aloss', verbose=1,
                     save_best_only=False, mode='min')

hist = model.fit_generator(epochs=10,
                            generator=train_gen,
                            validation_data=val_gen,
                            use_multiprocessing=True,
                            callbacks=[es, cp],
                            workers=-1)
#%% Plot curves
plt.figure(figsize=(10, 14))
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# %%
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

predictions = model.predict_generator(test_gen, use_multiprocessing=True,
                            workers=-2)
print('Confusion Matrix')
print(confusion_matrix(test_gen.get_labels(), np.argmax(predictions,axis=1)))
print('Classification Report')
print(classification_report(test_gen.get_labels(), np.argmax(predictions,axis=1), target_names=target_names))
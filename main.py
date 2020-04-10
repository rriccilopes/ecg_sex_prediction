# -*- coding: utf-8 -*-
'''
Created on Thu Apr  9 11:31:29 2020

@author: rriccilopes
'''

# Load data
import os
from matplotlib import pyplot as plt

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
from architectures import conv_net
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.applications.resnet50 import ResNet50
#model = ResNet50()
#model.summary()
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import pandas as pd

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

params = {'dim':(8, 2500), 'n_channels':1, 'n_classes':2,
          'batch_size':64, 'shuffle':True}

use_all_date = True

if use_all_date:
    train_gen = DataGenerator(partition['train'], labels,target_names=target_names, **params)
    val_gen = DataGenerator(partition['validation'], labels, target_names=target_names,**params)
    test_gen = DataGenerator(partition['test'], labels, target_names=target_names, **params)
else:
    import random
    train_gen = DataGenerator(random.sample(partition['train'], 64*6),
                              labels, target_names=target_names, **params)
    val_gen = DataGenerator(random.sample(partition['validation'], 64*3),
                            labels, target_names=target_names,**params)
    test_gen = DataGenerator(random.sample(partition['test'], 64*3),
                             labels, target_names=target_names, **params)

# Search for LR, get model and train network
for lr in [0.001, 0.0001, 0.00001]:
    model_path = 'models_' + str(lr) + '/'
    os.mkdir(model_path)
    
    model = conv_net()
    model.summary()
    
    # Define parameters
    optimizer = Adam(lr=lr)
    loss = 'categorical_crossentropy'
    metrics = ['accuracy', auroc]
    
    send_wpp_score = True
    epochs = 50
    
    # Call backs
    model_fp = model_path + "saved-model-{epoch:02d}-{val_loss:.3f}.hdf5"
    cp = ModelCheckpoint(model_fp, monitor='val_loss', verbose=1,
                         save_best_only=False, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    cb_list = [es, cp]
    
    if send_wpp_score:
        from whatsapp import NotifyWhatsAppCallback
        cb_list.append(NotifyWhatsAppCallback())
    
    # Train model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    hist = model.fit_generator(epochs=epochs,
                                generator=train_gen,
                                validation_data=val_gen,
                                use_multiprocessing=False,
                                callbacks=cb_list,
                                workers=-1)

    # Save history to csv    
    pd.DataFrame.from_dict(hist.history).round(4).to_csv(model_path + 'history.csv', index=False)
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from keras import models 
import numpy as np
import tensorflow as tf

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


model = models.load_model('models/saved-model-02-0.72.hdf5',  custom_objects={'auroc': auroc})

predictions = model.predict_generator(test_gen, use_multiprocessing=False,
                            workers=1)
print('Confusion Matrix')
print(confusion_matrix(test_gen.get_labels(), np.argmax(predictions,axis=1)))
print('Classification Report')
print(classification_report(test_gen.get_labels(), np.argmax(predictions,axis=1), target_names=target_names))
print(roc_auc_score(test_gen.get_labels(), np.argmax(predictions,axis=1)))
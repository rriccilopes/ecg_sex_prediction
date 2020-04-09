# -*- coding: utf-8 -*-
from keras import layers, models

def conv_net():
    """
    Adapted from https://www.ahajournals.org/doi/full/10.1161/CIRCEP.119.007284
    (Reported results) Accuracy on validation = 90.4%
    AUC on validation = 0.973, AUC on test = 0.968
    """
    
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
    # Using 8 leads instead of (original) 12
    model.add(layers.Conv2D(128, (8, 1), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((1, 2)))
    
    # FC Layers
    for n in [128, 64]:
        model.add(layers.Dense(n, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        # Dropout % not available
        model.add(layers.Dropout(0.5))
    
    # Output
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation="softmax"))
    
    return model
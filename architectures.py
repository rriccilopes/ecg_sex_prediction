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


def resnet(cardinality=1):
    """
    Adapted from  https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
    
    ** NOT TESTED YET **
    """
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y
    def residual_network(x):
        # conv1
        x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
        x = add_common_layers(x)
    
        # conv2
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        for i in range(3):
            project_shortcut = True if i == 0 else False
            x = residual_block(x, 128, 256, _project_shortcut=project_shortcut)
    
        # conv3
        for i in range(4):
            # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
            strides = (2, 2) if i == 0 else (1, 1)
            x = residual_block(x, 256, 512, _strides=strides)
    
        # conv4
        for i in range(6):
            strides = (2, 2) if i == 0 else (1, 1)
            x = residual_block(x, 512, 1024, _strides=strides)
    
        # conv5
        for i in range(3):
            strides = (2, 2) if i == 0 else (1, 1)
            x = residual_block(x, 1024, 2048, _strides=strides)
    
        x = layers.GlobalAveragePooling2D()(x)
    
        # x = layers.Dense(1)(x)
        x = layers.Dense(2, activation="softmax")(x)
        
        return x

    # Create model
    input_tensor = layers.Input(shape=(8, 2500, 1))
    network_output = residual_network(input_tensor)
    
    model = models.Model(inputs=[input_tensor], outputs=[network_output])
    
    return model
import numpy as np
import os
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from dice_coefficient_loss import dice_coefficient_loss, dice_coefficient
from metric import *


#TODO make BVNet static like unet
def BVNet(pretrained_weights = None,input_size = (256,256, 5)):
    # Build U-Net model
    inputs = Input((input_size))
   # s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(64, (3, 3), padding='same') (inputs)
    #c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
   # c1 = Dropout(0.1) (c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same') (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(128, (3, 3), padding='same') (c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2), strides=(2,2)) (c1)

#Encode block 2
    c2 = Conv2D(128, (3, 3), padding='same') (p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    #c2 = Dropout(0.1) (c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same') (c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(256, (3, 3), activation='relu', padding='same') (c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2), strides = (2,2)) (c2)

#Encode block 3
    c3 = Conv2D(256, (3, 3), padding='same') (p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    #c3 = Dropout(0.2) (c3)
    c3 = Conv2D(256, (3, 3), padding='same') (c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(512, (3, 3), padding='same') (c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2), strides=(2,2)) (c3)

# Encode block 4
    c4 = Conv2D(512, (3, 3), padding='same') (p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    #c4 = Dropout(0.2) (c4)
    c4 = Conv2D(512, (3, 3),  padding='same') (c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Conv2D(1024, (3, 3),  padding='same') (c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)

#Decode block 1
    u7 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (c4)
    u7 = BatchNormalization()(u7)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(512, (3, 3), padding='same') (u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    #c7 = Dropout(0.2) (c7)
    c7 = Conv2D(512, (3, 3),  padding='same') (c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv2D(512, (3, 3),  padding='same') (c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)

#Decode block 2
    u8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = BatchNormalization()(u8)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(256, (3, 3), padding='same') (u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)
    c8 = Conv2D(256, (3, 3), padding='same') (u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)
    #c8 = Dropout(0.1) (c8)
    c8 = Conv2D(256, (3, 3), padding='same') (c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)

#Decode block 3
    u9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(128, (3, 3), padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same') (u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)

#c9 = Dropout(0.1) (c9)
    #c9 = Conv2D(64, (3, 3), padding='same') (c9)
    #c9 = BatchNormalization()(c9)
    #c9 = Activation('relu')(c9)

    outputs = Conv2D(1, (1, 1), activation='softmax') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, decay=0.02), loss=dice_coefficient_loss, metrics=[binary_accuracy, dice_coefficient, recall, precision])
    model.summary()
    return model


def unet(pretrained_weights=None, input_size=(256, 256, 1), number_of_classes=1, loss_function="binaray_crossentropy"):
    inputs = Input(input_size)

    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu',
                padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu',
                padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu',
                padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu',
                padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu',
                padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu',
                padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu',
                padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(1024, (3, 3), activation='relu',
                padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu',
                padding='same')(c5)

    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu',
                padding='same')(u6)
    c6 = Conv2D(256, (3, 3), activation='relu',
                padding='same')(c6)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu',
                padding='same')(u7)

    c7 = Conv2D(128, (3, 3), activation='relu',
                padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu',
                padding='same')(u8)

    c8 = Conv2D(128, (3, 3), activation='relu',
                padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(64, (3, 3), activation='relu',
                padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu',
                padding='same')(c9)

    outputs = Conv2D(number_of_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=SGD(lr=10e-5), loss=dsc_loss, metrics=[binary_accuracy, dsc, recall, precision])
    model.summary()
    return model

if __name__ == "__main__":
    unet()

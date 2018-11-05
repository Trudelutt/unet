import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.misc
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN
from unet import unet
from preprossesing import get_training_data
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def gpu_config():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    set_session(tf.Session(config = config))

def write_png(path, array):
    new_array = array.reshape(512,512)
    scipy.misc.imsave(path, new_array)

def visulize_predic(p):
    new_p = np.zeros((p.shape[0], p.shape[1]))
    print("NEW P")
    print(new_p.shape)
    for i in range(new_p.shape[0]):
        for j in range(new_p.shape[1]):
            if(p[i][j][0] < p[i][j][1]):
                new_p[i][j] = 1
    return new_p

def train_model(model, input, label):
    print("Training sample" + str(new_x_train.shape))
    model_checkpoint = ModelCheckpoint('unet_vessels.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model_earlyStopp = EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=1, mode='min', baseline=None, restore_best_weights=False)
    model.fit(x=input, y= label, batch_size=1, epochs=70, verbose=1, callbacks=[model_checkpoint, model_earlyStopp, TerminateOnNaN()])

def predict_model(model, input):
    print("Starting predictions")
    p = model.predict(input)
    print("Writing predictions to file...")
    write_png("./predictions/org20.png", new_x_train[20])

    #write_png("./predictions/background.png", p[0][...,0])
    write_png("./predictions/prediction20.png", p[0][...,1])
    write_png("./predictions/predictiontumor5.png", p[0][...,2])

    #write_png("./predictions/backgroundgt.png", one_hot_label[0][...,0])
    #write_png("./predictions/gt.png", one_hot_label[0][...,0])
    #write_png("./predictions/tumorgt.png", one_hot_label[0][...,1])



gpu_config()
model = unet()
train, label = get_training_data()
one_hot_label = to_categorical(label, num_classes=3)


new_x_train = train.reshape(train.shape[0], train.shape[1], train.shape[2], 1)
print("Training sample" + str(new_x_train.shape))
train_model(model, new_x_train, one_hot_label)

one_sample = new_x_train[20:21]
predict_model(model, one_sample)

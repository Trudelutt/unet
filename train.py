import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.misc
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN
from keras.models import load_model
from unet import unet
from preprossesing import get_training_data
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from metric import mean_iou


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

def train_model(model, input, target):
    print("Inside training")
    print("Training sample" + str(input.shape))
   # print("label sample" + str(target))

    model_checkpoint = ModelCheckpoint('unet_vessels.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model_earlyStopp = EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=1, mode='min', baseline=None, restore_best_weights=False)
    model.fit(x=input, y= target, batch_size=1, epochs=100, verbose=1, callbacks=[model_checkpoint, model_earlyStopp, TerminateOnNaN()])

def predict_model(model, input, target):
    print("Starting predictions")
    p = model.predict(input)
    print("prediction: " + str(p))
    write_predictions_to_file(p, target)
    #show_predictions(p, target)


def show_predictions(prediction_array, mask_array):
    plt.figure()
    print(prediction_array.shape)
    plt.imshow(prediction_array[0][...,0], cmap='gray')
    plt.figure()
    plt.imshow(prediction_array[0], cmap='gray')
    plt.figure()
    plt.imshow(prediction_array[0][...,1], cmap='gray')
    plt.figure()
    plt.imshow(prediction_array[0][...,2], cmap='gray')
    plt.figure()
    plt.imshow(mask_array[0][...,1], cmap='gray')
    plt.show()

def write_predictions_to_file(p, target):
    print("Writing predictions to file...")
    write_png("./predictions/gt20.png", target[0][...,0])

    #write_png("./predictions/background.png", p[0][...,0])
    write_png("./predictions/prediction20.png", p[0][...,0])
    #write_png("./predictions/predictiontumor5.png", p[0][...,2])

if __name__ == "__main__":
    gpu_config()
    model = unet()
    train, label = get_training_data()
    one_hot_label = to_categorical(label, num_classes=3)[...,1:-1]


    new_x_train = train.reshape(train.shape[0], train.shape[1], train.shape[2], 1)
    print("Training sample" + str(new_x_train.shape))
    train_model(model, new_x_train[20:21], one_hot_label[20:21])
    #pre_train_model = load_model("unet_vessels.hdf5", custom_objects={'mean_iou': mean_iou})
    one_sample = new_x_train[20:21]
    predict_model(model, one_sample, one_hot_label[20:21])

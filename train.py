import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.misc
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN
from keras.models import load_model
from model import unet, BVNet
from preprossesing import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from metric import mean_iou
from dice_coefficient_loss import dice_coefficient_loss


def gpu_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    #set_session(tf.Session(config = config))
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())



def write_png(path, array):
    #new_array = array.reshape(512,256)
    scipy.misc.imsave(path, array)

def visulize_predic(p):
    new_p = np.zeros((p.shape[0], p.shape[1]))
    print("NEW P")
    print(new_p.shape)
    for i in range(new_p.shape[0]):
        for j in range(new_p.shape[1]):
            if(p[i][j][0] < p[i][j][1]):
                new_p[i][j] = 1
    return new_p

def train_model(model, input, target, val_x, val_y):
    print("Inside training")
    print("Training sample" + str(input.shape))
   # print("label sample" + str(target))

    model_checkpoint = ModelCheckpoint('unet_vessels.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model_earlyStopp = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=1, mode='min', baseline=None, restore_best_weights=False)
    model.fit(x=input, y= target, validation_data=(val_x, val_y), batch_size=1, epochs=1, verbose=1, callbacks=[model_checkpoint, model_earlyStopp, TerminateOnNaN()])

def predict_model(model, input, target):
    print("Starting predictions")
    p = model.predict(input)
    print("prediction: " + str(p))
    write_pridiction_to_file(target,p, path="./predictions/prediction.nii.gz")


def show_predictions(prediction_array, mask_array):
    plt.figure()
    print(prediction_array.shape)
    plt.imshow(prediction_array[...,2], cmap='gray')
    """plt.figure()
    plt.imshow(prediction_array[0], cmap='gray')
    plt.figure()
    plt.imshow(prediction_array[0][...,1], cmap='gray')
    plt.figure()
    plt.imshow(prediction_array[0][...,2], cmap='gray')"""
    plt.figure()
    plt.imshow(mask_array[...,0], cmap='gray')
    plt.show()

def write_predictions_to_pngfile(p, target):
    print("Writing predictions to file...")
    write_png("./predictions/gt20.png", target[0][...,0])

    #write_png("./predictions/background.png", p[0][...,0])
    write_png("./predictions/prediction20.png", p[0][...,0])
    #write_png("./predictions/predictiontumor5.png", p[0][...,2])

if __name__ == "__main__":
    overwrite = True
    gpu_config()
    model = BVNet()

    train_files, val_files, test_files = get_data_files(data="ca", label="LM")
    train_data, label_data = get_train_data_slices(train_files[:1])
    val_data, val_label = get_slices(val_files[:1])

    #one_hot_label = to_categorical(label, num_classes=3)[...,1:-1]
    #show_predictions(train[20], one_hot_label[20])
    #new_x_train = train.reshape(train.shape[0], train.shape[1], train.shape[2], 1)
    #print("Training sample" + str(train_data.shape))
    if  not overwrite:
        prediction_model= load_model("unet_vessels.hdf5", custom_objects={'mean_iou': mean_iou,  'dice_coefficient_loss': dice_coefficient_loss})
    else:
        train_model(model, train_data, label_data, val_data, val_label)
        prediction_model = model
    pred_sample, pred_label = get_prediced_image_of_test_files(test_files, 0)
    predict_model(prediction_model, pred_sample, pred_sample)

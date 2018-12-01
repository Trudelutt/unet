import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.misc
import json
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN
from keras.models import load_model
from model import unet, BVNet
from preprossesing import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from metric import *
from dice_coefficient_loss import dice_coefficient_loss


def gpu_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    #set_session(tf.Session(config = config))
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())


def train_model(model, input, target, val_x, val_y, modelpath):
    print("Inside training")

    model_checkpoint = ModelCheckpoint("./models/"+ modelpath +".hdf5", monitor='val_loss',verbose=1, save_best_only=True)
    model_earlyStopp = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=1, mode='min', baseline=None, restore_best_weights=False)
    history = model.fit(x=input, y= target, validation_data=(val_x, val_y), batch_size=1, epochs=1, verbose=1, callbacks=[model_checkpoint, model_earlyStopp, TerminateOnNaN()])
    with open('./history/'+ modelpath + '.json', 'w') as f:
        json.dump(history.history, f)
        print("Saved history....")

def predict_model(model, input, target, name='LM_01', label="LM"):
    print("Starting predictions")
    p = model.predict(input,  batch_size=1, verbose=1)
    write_pridiction_to_file(target,p, path="./predictions/" +label + "/" + name + "prediction.nii.gz")


def evaluation(model, test_files):
    test_x, test_y = get_train_data_slices(test_files)
    print("Starting evaluation.....")
    print(model.evaluate(test_x, test_y), batch_size=1, verbose=1)
    print(model.metrics_names)
    print("Evaluation done..")


if __name__ == "__main__":
    overwrite = True
    gpu_config()
    model_name = "BVNet"
    #Hepatic Vessel has label HV
    label = "LM"
    modelpath = model_name+ "_"+ label
    custom_objects = custom_objects={ 'binary_accuracy':binary_accuracy, 'recall':recall,
    'precision':precision, 'dsc': dsc, 'dsc_loss': dsc_loss}
    if model_name == "BVNet":
        model = BVNet()
    else:
        model_name="unet"
        model = unet()

    train_files, val_files, test_files = get_data_files(data="ca", label=label)
    train_data, label_data = get_train_data_slices(train_files)
    print("Done geting training slices...")
    val_data, val_label = get_slices(val_files)
    print("Done geting validation slices...")

    if  not overwrite:
        prediction_model= load_model('./models/' + modelpath +'.hdf5', custom_objects=custom_objects)
    else:
        train_model(model, train_data, label_data, val_data, val_label, modelpath=modelpath)
        prediction_model = load_model('./models/' + modelpath +'.hdf5', custom_objects=custom_objects)
    for i in range(len(test_files)):
        pred_sample, pred_label = get_prediced_image_of_test_files(test_files, i)
        predict_model(prediction_model, pred_sample, pred_sample, name=modelpath+"_"+str(i)+"_", label=label)
    evaluation(prediction_model, test_files)

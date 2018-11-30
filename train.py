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
from metric import *
from dice_coefficient_loss import dice_coefficient_loss
from pickle import dump, load


def gpu_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    #set_session(tf.Session(config = config))
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())




"""def write_png(path, array):
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

def write_predictions_to_pngfile(p, target, path):
    print("Writing predictions to file...")
    write_png(path, target[0][...,0])
    write_png("./predictions/prediction20.png", p[0][...,0])"""


def plot_history(history, name, save=False):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(name+' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save:
        plt.savefig('fig/'+name+'_acc.png')
        plt.close()
    else:
        plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(name+' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save:
        plt.savefig('fig/'+name+'_loss.png')
        plt.close()
    else:
        plt.show()

def train_model(model, input, target, val_x, val_y, modelpath):
    print("Inside training")
    print("Training sample" + str(input.shape))

    model_checkpoint = ModelCheckpoint("./models/"+ modelpath +".hdf5", monitor='val_loss',verbose=1, save_best_only=True)
    model_earlyStopp = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=1, mode='min', baseline=None, restore_best_weights=False)
    history = model.fit(x=input, y= target, validation_data=(val_x, val_y), batch_size=1, epochs=500, verbose=1, callbacks=[model_checkpoint, model_earlyStopp, TerminateOnNaN()])
    dump(history, open('./history/'+modelpath + '_history.pkl', 'wb'))

def predict_model(model, input, target, name='LM_01'):
    print("Starting predictions")
    p = model.predict(input)
    print("prediction: " + str(p))
    print(p.shape, target.shape)
    write_pridiction_to_file(target,p, path="./predictions/" + name + "prediction.nii.gz")


def evaluation(model, test_files):
    test_x, test_y = get_train_data_slices(train_files)
    print("Starting evaluation.....")
    print(model.evaluate(test_x, test_y))
    print("Evaluation done..")


if __name__ == "__main__":
    overwrite = False
    gpu_config()
    model_name = "BVNet"
    label = "LM"
    modelpath = model_name+ "_"+ label
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
        prediction_model= load_model(modelpath, custom_objects={'mean_iou': mean_iou, 'accuracy':accuracy, 'recall':recall,
        'precision':precision, 'dsc': dsc, 'dsc_loss': dsc_loss})
    else:
        train_model(model, train_data, label_data, val_data, val_label, modelpath=modelpath)
        prediction_model = load_model(modelpath, custom_objects={'mean_iou': mean_iou, 'accuracy':accuracy, 'recall':recall,
        'precision':precision, 'dsc': dsc, 'dsc_loss': dsc_loss})
    for i in range(len(test_files)):
        pred_sample, pred_label = get_prediced_image_of_test_files(test_files, i)
        #print(pred_sample)
        predict_model(prediction_model, pred_sample, pred_sample, name=model_path+"_"+i+"_")
    evaluation(prediction_model, test_files)

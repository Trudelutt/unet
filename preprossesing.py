import os
import json
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm



def split_train_val_test(files):
    n_files = len(files)
    number_test = n_files//((n_files - int(n_files*0.8))//2)

    val_files = files[::number_test]
    test_files = files[1::number_test]
    train_files = [x for x in files if x not in val_files and x not in test_files]
    print(n_files, len(train_files), len(val_files), len(test_files))
    if(bool(set(val_files) & set(test_files) & set(train_files))):
        print("THEY have some element in common")

    return train_files, val_files, test_files

#TODO check if any information is lost here
def preprosses_images(image, label):
    image -= np.min(image)
    image = image/ np.max(image)
    image -= np.mean(image)
    image = image / np.std(image)
    return image[:,128:-128,128:-128], label[:,128:-128,128:-128]

def remove_slices_with_just_background(image, label):
    first_non_backgroud_slice = float('inf')
    last_non_backgroud_slice = -1
    for i in range(image.shape[0]):
        if(1 in label[i]):
            if(i < first_non_backgroud_slice):
                first_non_backgroud_slice = i
            last_non_backgroud_slice = i
    resize_label =  label[first_non_backgroud_slice - 2 - 2 :last_non_backgroud_slice + 1 + 2 + 2]
    resize_image =  image[first_non_backgroud_slice-2:last_non_backgroud_slice+1+2]
    return resize_image, resize_label

def add_neighbour_slides_training_data(image, label):
    image_with_channels = np.zeros((image.shape[0]-4, 256, 256, 5))
    for i in range(image_with_channels.shape[0]):
        if(i < image.shape[0]):
            image_with_channels[i][...,0] = image[i]
        if(i+1 < image.shape[0]):
            image_with_channels[i][...,1] = image[i+1]
        if(i+2 < image.shape[0]):
            image_with_channels[i][...,2] = image[i+2]
        if(i+3 < image.shape[0]):
            image_with_channels[i][...,3] = image[i+3]
        if(i+4 < image.shape[0]):
            image_with_channels[i][...,4] = image[i+4]

    #TODO check if channels becomes right for training 0. 1. and the last ones
    """if np.array_equal(image_with_channels[20][...,0],image[20]):
        print("HURRA channel 0 er riktig")
    if np.array_equal(image_with_channels[20][...,1], image[21]):
        print("HURRA channel 1 er riktig")
    if np.array_equal(image_with_channels[20][...,2], image[22]):
        print("HURRA channel 2 er riktig")
    if np.array_equal(image_with_channels[20][...,3], image[23]):
        print("HURRA channel 3 er riktig")
    if np.array_equal(image_with_channels[20][...,4], image[24]):
        print("HURRA channel 4 er riktig")"""

    return image_with_channels, label

def fetch_training_data_hapaticv_files():
    training_data_files = list()
    path = "../Task08_HepaticVessel/"
    count = 0
    with open("../Task08_HepaticVessel/dataset.json", "r+", encoding="utf-8") as f:
        data = json.load(f)
        for i in range(len(data["training"])):
            subject_files = list()
            subject_files.append(path + data["training"][i]["image"].replace("./", ""))
            subject_files.append(path + data["training"][i]["label"].replace("./", ""))
            training_data_files.append(tuple(subject_files))
    return training_data_files

def fetch_training_data_ca_files(label="LM"):
    path = glob("../st.Olav/*/*/*/")
    training_data_files = list()
    for i in range(len(path)):
        try:
            data_path = glob(path[i] + "*CCTA.nii.gz")[0]
            label_path = glob(path[i] + "*" + label + ".nii.gz")[0]
        except IndexError:
            print("out of range for %s" %(path[i]))
        else:
            training_data_files.append(tuple([data_path, label_path]))
    return training_data_files

def get_preprossed_numpy_arrays_from_file(image_path, label_path):
    sitk_image  = sitk.ReadImage(image_path)
    sitk_label  = sitk.ReadImage(label_path)
    return preprosses_images(sitk.GetArrayFromImage(sitk_image), sitk.GetArrayFromImage(sitk_label))

#TODO fiks so indexes
def get_train_and_label_numpy(number_of_slices, train_list, label_list):
    train_data = np.zeros((number_of_slices, train_list[0].shape[1], train_list[0].shape[2], 5))
    label_data = np.zeros((number_of_slices, label_list[0].shape[1], label_list[0].shape[2]))
    index = 0
    for i in range(len(train_list)):
        #print("image " + str(i+1) +"/" + str(len(train_list)))
        with tqdm(total=train_list[i].shape[0], desc='Adds splice  from image ' + str(i+1) +"/" + str(len(train_list))) as t:
            for k in range(train_list[i].shape[0]):
                #print(str(index+1) +"/" + str(train_data.shape[0]))
                train_data[index] = train_list[i][k]
                label_data[index] = label_list[i][k]
                index += 1
                t.update()

    return train_data, label_data


def read_numpyarray_from_file(path):
    image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image)

def show_nii_image(path, slice_nr):
    image = read_numpyarray_from_file(path)
    plt.figure()
    plt.imshow(image[slice_nr])

#TODO make sure that index not out of bounds
def get_prediced_image_of_test_files(files, number):
    element = files[number]
    print("Prediction on " + element[0])
    return get_slices(files[number:number+1])


def write_pridiction_to_file(image_array, prediction_array, path="./predictions/prediction.nii.gz"):
    print(prediction_array.shape)
    sitk_image = sitk.GetImageFromArray(image_array)
    sitk.WriteImage(sitk_image, path.replace("prediction.nii", "gt.nii"))
    predsitk_image = sitk.GetImageFromArray(prediction_array)
    sitk.WriteImage(predsitk_image, path)
    print("Writing prediction is done...")




# Assume to have some sitk image (itk_image) and label (itk_label)
def get_data_files(data="ca", label="LM"):
    if data == "ca":
        files = fetch_training_data_ca_files(label)
    else:
        files = fetch_training_data_hapaticv_files()

    print("files: " + str(len(files)))
    return split_train_val_test(files)


def get_train_data_slices(train_files):
    traindata = []
    labeldata = []
    count_slices = 0
    for element in train_files:
        print(element[0])
        numpy_image, numpy_label = get_preprossed_numpy_arrays_from_file(element[0], element[1])
        resized_image, resized_label = remove_slices_with_just_background(numpy_image, numpy_label)
        i, l = add_neighbour_slides_training_data(resized_image, resized_label)
        count_slices += i.shape[0]
        traindata.append(i)
        labeldata.append(l)
    train_data, label_data = get_train_and_label_numpy(count_slices, traindata, labeldata)

    print("min: " + str(np.min(train_data)) +", max: " + str(np.max(train_data)))
    label = label_data.reshape((label_data.shape[0], label_data.shape[1], label_data.shape[2], 1))
    return train_data, label

def get_slices(files):
    input_data_list = []
    label_data_list = []
    count_slices = 0
    for element in files:
        numpy_image, numpy_label = get_preprossed_numpy_arrays_from_file(element[0], element[1])
        i, l = add_neighbour_slides_training_data(numpy_image, numpy_label)
        count_slices += i.shape[0]
        input_data_list.append(i)
        label_data_list.append(l)
    train_data, label_data = get_train_and_label_numpy(count_slices, input_data_list, label_data_list)

    print("min: " + str(np.min(train_data)) +", max: " + str(np.max(train_data)))
    label = label_data.reshape((label_data.shape[0], label_data.shape[1], label_data.shape[2], 1))
    return train_data, label


if __name__ == "__main__":
    train_files, val_files, test_files = get_data_files(data="ca", label="Aorta")
    #print("#####")
    #print(val_files)
    #print("#####")
    #print(test_files)
    #print("#####")
    #print(train_files)
    train_data, label_data = get_train_data_slices(train_files[:1])
    print(train_data)
    #write_pridiction_to_file(train_data[...,2], label_data)

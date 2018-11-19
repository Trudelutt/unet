import os
import json
import SimpleITK as sitk
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# A path to a T1-weighted brain .nii image:
#t1_fn = '../Task02_Heart/imagesTr/la_003.nii'

# Read the .nii image containing the volume with SimpleITK:
#sitk_t1 = sitk.ReadImage(t1_fn)
#itk_label = sitk.ReadImage('../Task02_Heart/labelsTr/la_003.nii')
# and access the numpy array:
#t1 = sitk.GetArrayFromImage(sitk_t1)

"""def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):

    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    out_size = [ 256,256, original_size[2]]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)"""

def split_train_val_test(files):
    n_files = len(files)
    #print(n_files)
    sep_train = int(n_files*0.6)
    #print(sep_train)
    sep_val = int(n_files*0.8)
    n_train = sep_train
    n_val = sep_val - sep_train
    n_test = n_files - n_train - n_val

    train_files = files[:sep_train]
    val_files = files[sep_train:sep_val]

    test_files = files[sep_val:]

    return train_files, val_files, test_files

#TODO check if any information is lost here
def preprosses_images(image, label):
    image -= np.min(image)
    image = image/ np.max(image)
    image -= np.mean(image)
    image = image / np.std(image)
    return image[:,128:-128,128:-128], label[:,128:-128,128:-128]

def remove_slices_with_just_background(image, label):
    #print("IN SCOPE")
    #print(image)
    first_non_backgroud_slice = float('inf')
    last_non_backgroud_slice = -1
    for i in range(image.shape[0]):
        if(1 in label[i]):
            if(i < first_non_backgroud_slice):
                first_non_backgroud_slice = i
            last_non_backgroud_slice = i
    #print("Image size " + str(image.shape))
    resize_label =  label[first_non_backgroud_slice:last_non_backgroud_slice+1]
    resize_image =  image[first_non_backgroud_slice:last_non_backgroud_slice+1]
    return resize_image, resize_label

def add_neighbour_slides_training_data(image, label):
    image_with_channels = np.zeros((image.shape[0], 256, 256, 5))
    #print("KILLED AFTER channels")
    for i in range(image.shape[0]):
        if(i-2 >= 0):
            image_with_channels[i][...,0] = image[i-2]
        #else:
            #image_with_channels[i][...,0] = image[first_non_backgroud_slice-1:first_non_backgroud_slice,128:-128,156:-100]
        if(i-1>= 0):
            image_with_channels[i][...,1] = image[i-1]
        #else:
        #    image_with_channels[i][...,1] = image[first_non_backgroud_slice-2:first_non_backgroud_slice - 1,128:-128,156:-100]
        image_with_channels[i][...,2] = image[i]
        if(i+1 < image.shape[0]):
            image_with_channels[i][...,3] = image[i+1]
        #else:
            #image_with_channels[i][...,3] = image[last_non_backgroud_slice + 1:last_non_backgroud_slice +2,128:-128,156:-100]
        if(i+2 < image.shape[0]):
            #print("IT HAS not")
            image_with_channels[i][...,4] = image[i+2]
        #else:
            #print(resize_image.shape, image.shape)
            #print("EDGE case")
            #image_with_channels[i][...,4] = image[last_non_backgroud_slice + 2:last_non_backgroud_slice +3,128:-128,156:-100]
    #TODO check if channels becomes right for training 0. 1. and the last ones
    """if np.array_equal(image_with_channels[20][...,0],resize_image[18]):
        print("HURRA channel 0 er riktig")
    if np.array_equal(image_with_channels[20][...,1], resize_image[19]):
        print("HURRA channel 1 er riktig")
    if np.array_equal(image_with_channels[20][...,2], resize_image[20]):
        print("HURRA channel 2 er riktig")
    if np.array_equal(image_with_channels[20][...,3], resize_image[21]):
        print("HURRA channel 3 er riktig")
    if np.array_equal(image_with_channels[20][...,4], resize_image[22]):
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
    data_path = glob("../st.Olav/*/*/*/*CCTA.nii.gz")
    label_path = glob("../st.Olav/*/*/*/*"+label+".nii.gz")
    training_data_files = list()
    for i in range(len(data_path)):
        training_data_files.append(tuple([data_path[i], label_path[i]]))
    return training_data_files

def get_preprossed_numpy_arrays_from_file(image_path, label_path):
    sitk_image  = sitk.ReadImage(image_path)
    sitk_label  = sitk.ReadImage(label_path)
    print(image_path)
    #print(sitk.GetArrayFromImage(sitk_image))
    return preprosses_images(sitk.GetArrayFromImage(sitk_image), sitk.GetArrayFromImage(sitk_label))

#TODO fiks so indexes
def get_train_and_label_numpy(number_of_slices, train_list, label_list):
    train_data = np.zeros((number_of_slices, train_list[0].shape[1], train_list[0].shape[2], 5))
    label_data = np.zeros((number_of_slices, label_list[0].shape[1], label_list[0].shape[2]))
    index = 0
    for i in range(len(train_list)):
        print("image " + str(i+1) +"/" + str(len(train_list)))

        for k in range(train_list[i].shape[0]):
            print(str(index+1) +"/" + str(train_data.shape[0]))
            #print("slices of image " + str(k+1) +"/" + str(train_list[j].shape[0]))
            #print(train_list[j][k])
            train_data[index] = train_list[i][k]
            label_data[index] = label_list[i][k]
            index += 1

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
        #resized_image, resized_label = remove_slices_with_just_background(numpy_image, numpy_label)
        i, l = add_neighbour_slides_training_data(numpy_image, numpy_label)
        count_slices += i.shape[0]
        input_data_list.append(i)
        label_data_list.append(l)
    train_data, label_data = get_train_and_label_numpy(count_slices, input_data_list, label_data_list)

    print("min: " + str(np.min(train_data)) +", max: " + str(np.max(train_data)))
    label = label_data.reshape((label_data.shape[0], label_data.shape[1], label_data.shape[2], 1))
    return train_data, label


if __name__ == "__main__":
    train_files, val_files, test_files = get_data_files(data="ca", label="LM")
    train_data, label_data = get_train_data_slices(train_files[:1])
    print(train_data)
    #write_pridiction_to_file(train_data[...,2], label_data)

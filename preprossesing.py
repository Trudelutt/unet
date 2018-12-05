import os
import json
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize



def split_train_val_test(samples):
    n_samples = len(samples)
    sep_train = int(n_samples*0.8)
    sep_val = int(n_samples*0.9)

    train_files = samples[:sep_train]
    val_files = samples[sep_train:sep_val]

    test_files = samples[sep_val:]
    print(len(train_files), len(val_files), len(test_files))
    return train_files, val_files, test_files

#TODO check if any information is lost here
def preprosses_images(image, label, tag):
    #image = resize(image, (image.shape[0], 256, 256))
    image -= np.min(image)
    image = image/ np.max(image)
    image -= np.mean(image)
    image = image / np.std(image)
    #label = resize(label, (label.shape[0],256, 256))
    print(np.unique(label))

    if tag == "HV":
        print(image[:,:-128,200:-56].shape)
        return image[:,128:-128,200:-56], label[:,128:-128,200:-56]
    else:
        return image[:,100:-156, 80:-176], label[:,100:-156, 80:-176]
    """elif tag == "Aorta":
        return image[:], label[:]
    else:
        return image[:,128:-128,128:-128], label[:,128:-128,128:-128]"""


def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):

    # The image we will resample (a grid).
    """grid_image = sitk.GridSource(outputPixelType=sitk.sitkUInt16, size=(512,512),
                                 sigma=(0.1,0.1), gridSpacing=(20.0,20.0))
    sitk.Show(grid_image, "original grid image")"""

    # The spatial definition of the images we want to use in a deep learning framework (smaller than the original).
    #new_size = [256, 256, itk_image.GetDepth()]
    #reference_image = sitk.Image(new_size, itk_image.GetPixelIDValue())
    #reference_image.SetOrigin(itk_image.GetOrigin())
    #reference_image.SetDirection(itk_image.GetDirection())
    #resample.SetSpacing([sz*spc/nsz for nsz,sz,spc in zip(new_size, grid_image.GetSize(), grid_image.GetSpacing())])

    # Resample without any smoothing.
    #sitk.Show(sitk.Resample(grid_image, reference_image) , "resampled without smoothing")

    # Resample after Gaussian smoothing.
    #sitk.Show(sitk.Resample(sitk.SmoothingRecursiveGaussian(grid_image, 2.0), reference_image), "resampled with smoothing")

    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    """out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]"""
    out_size = [ 256, 256, itk_image.GetDepth()]

    resample = sitk.ResampleImageFilter()
    #resample.SetOutputSpacing(original_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetOutputSpacing([1, 1, 1])

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def get_preprossed_numpy_arrays_from_file(image_path, label_path, tag):
    sitk_image  = sitk.ReadImage(image_path)
    sitk_label  = sitk.ReadImage(label_path)
    numpy_image = sitk.GetArrayFromImage(sitk_image)
    numpy_label = sitk.GetArrayFromImage(sitk_label)
    #print("NUmpy hsape " + str(numpy_image.shape))

    return preprosses_images(numpy_image, numpy_label, tag)

def remove_slices_with_just_background(image, label):
    first_non_backgroud_slice = float('inf')
    last_non_backgroud_slice = -1
    image_list = []
    label_list = []
    #print(image.shape)
    for i in range(image.shape[0]):
        #print(np.unique(label[i]), i)
        #print(label[i].shape)
        if(1 in label[i]):
            #print(i)
            if(i < first_non_backgroud_slice):
                first_non_backgroud_slice = i
            last_non_backgroud_slice = i
    #print(first_non_backgroud_slice-2, last_non_backgroud_slice)
    if(first_non_backgroud_slice-2 < 0):
        resize_label =  label[first_non_backgroud_slice-1:last_non_backgroud_slice + 1]
        resize_image =  image[first_non_backgroud_slice-1:last_non_backgroud_slice+1]
    elif(first_non_backgroud_slice-1 < 0):
        resize_label =  label[first_non_backgroud_slice:last_non_backgroud_slice + 1]
        resize_image =  image[first_non_backgroud_slice:last_non_backgroud_slice+1]
    else:
        resize_label =  label[first_non_backgroud_slice-2:last_non_backgroud_slice + 1]
        resize_image =  image[first_non_backgroud_slice-2:last_non_backgroud_slice+1]

    #print(resize_label.shape)
    #print(np.unique(resize_label[0]))
    return resize_image, resize_label


def add_neighbour_slides_training_data(image, label):
    image_with_channels = np.zeros((image.shape[0], image.shape[1], image.shape[2], 5))
    zeros_image = np.zeros(image[0].shape)
    for i in range(image.shape[0]):
        if(i == 0):
            image_with_channels[i][...,0] = zeros_image
            image_with_channels[i][...,1] = zeros_image
            image_with_channels[i][...,2] = image[i]
            image_with_channels[i][...,3] = image[i+1]
            image_with_channels[i][...,4] = image[i+2]
        elif(i == 1):
            image_with_channels[i][...,0] = zeros_image
            image_with_channels[i][...,1] = image[i-1]
            image_with_channels[i][...,2] = image[i]
            image_with_channels[i][...,3] = image[i+1]
            image_with_channels[i][...,4] = image[i+2]
        elif(i == image.shape[0]-2):
            image_with_channels[i][...,0] = image[i-2]
            image_with_channels[i][...,1] = image[i-1]
            image_with_channels[i][...,2] = image[i]
            image_with_channels[i][...,3] = image[i+1]
            image_with_channels[i][...,4] = zeros_image
        elif(i == image.shape[0]-1):
            image_with_channels[i][...,0] = image[i-2]
            image_with_channels[i][...,1] = image[i-1]
            image_with_channels[i][...,2] = image[i]
            image_with_channels[i][...,3] = zeros_image
            image_with_channels[i][...,4] = zeros_image
        else:
            image_with_channels[i][...,0] = image[i-2]
            image_with_channels[i][...,1] = image[i-1]
            image_with_channels[i][...,2] = image[i]
            image_with_channels[i][...,3] = image[i+1]
            image_with_channels[i][...,4] = image[i+2]
    return image_with_channels, label


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


def get_train_and_label_numpy(number_of_slices, train_list, label_list):
    train_data = np.zeros((number_of_slices, train_list[0].shape[1], train_list[0].shape[2], 5))
    label_data = np.zeros((number_of_slices, label_list[0].shape[1], label_list[0].shape[2]))
    index = 0
    for i in range(len(train_list)):
        with tqdm(total=train_list[i].shape[0], desc='Adds splice  from image ' + str(i+1) +"/" + str(len(train_list))) as t:
            for k in range(train_list[i].shape[0]):
                train_data[index] = train_list[i][k]
                label_data[index] = label_list[i][k]
                index += 1
                t.update()

    return train_data, label_data


def read_numpyarray_from_file(path):
    image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image).astype('float32')

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


def get_train_data_slices(train_files, tag = "LM"):
    traindata = []
    labeldata = []
    count_slices = 0
    for element in train_files:
        #if(element[0] == "../st.Olav/CT_FFR_Pilot_7/CT_FFR_Pilot_7_Segmentation/CT_FFR_Pilot_7_Segmentation_0000/CT_FFR_Pilot_7_Segmentation_0000_CCTA.nii.gz"):
        print(element[0])
        numpy_image, numpy_label = get_preprossed_numpy_arrays_from_file(element[0], element[1], tag)
        i, l = add_neighbour_slides_training_data(numpy_image, numpy_label)
        resized_image, resized_label = remove_slices_with_just_background(i, l)

        count_slices += resized_image.shape[0]
        traindata.append(resized_image)
        labeldata.append(resized_label)
        train_data, label_data = get_train_and_label_numpy(count_slices, traindata, labeldata)

    print("min: " + str(np.min(train_data)) +", max: " + str(np.max(train_data)))
    if tag == "HV":
        label_data = to_categorical(label_data,3)[:][...,1]
    label = label_data.reshape((label_data.shape[0], label_data.shape[1], label_data.shape[2], 1))
    return train_data, label


def get_slices(files, tag="LM"):
    input_data_list = []
    label_data_list = []
    count_slices = 0
    for element in files:
        numpy_image, numpy_label = get_preprossed_numpy_arrays_from_file(element[0], element[1],tag)
        i, l = add_neighbour_slides_training_data(numpy_image, numpy_label)
        count_slices += i.shape[0]
        input_data_list.append(i)
        label_data_list.append(l)
    train_data, label_data = get_train_and_label_numpy(count_slices, input_data_list, label_data_list)

    print("min: " + str(np.min(train_data)) +", max: " + str(np.max(train_data)))
    label = label_data.reshape((label_data.shape[0], label_data.shape[1], label_data.shape[2], 1))
    if tag == "HV":
        label = to_categorical(label,3)[:][...,1]
    return train_data, label


def write_all_labels(path):
    image = read_numpyarray_from_file(path+"LM.nii.gz")
    image += read_numpyarray_from_file(path+"Aorta.nii.gz")
    image += read_numpyarray_from_file(path+ "RCA.nii.gz")
    image[image == 2] = 1
    sitk_image = sitk.GetImageFromArray(image)
    sitk.WriteImage(sitk_image, "all_labels.nii.gz")

def data_augumentation(train_x, train_y):
    data_gen_args = dict( zoom_range=0.1)
    train_x1_datagen = ImageDataGenerator(**data_gen_args)
    train_x2_datagen = ImageDataGenerator(**data_gen_args)
    train_x3_datagen = ImageDataGenerator(**data_gen_args)
    train_x4_datagen = ImageDataGenerator(**data_gen_args)
    train_x5_datagen = ImageDataGenerator(**data_gen_args)
    train_y_datagen = ImageDataGenerator(**data_gen_args)
    seed = 1
    train_x1_augmented = train_x1_datagen.flow(train_x[...,0:1], batch_size=1, shuffle=False, seed=seed)
    #train_x2_augmented = train_x1_datagen.flow(train_x[...,1:2], batch_size=1, shuffle=False, seed=seed)
    #train_x3_augmented = train_x1_datagen.flow(train_x[...,2:3], batch_size=1, shuffle=False, seed=seed)
    #train_x4_augmented = train_x1_datagen.flow(train_x[...,3:4], batch_size=1, shuffle=False, seed=seed)
    #train_x5_augmented = train_x1_datagen.flow(train_x[...,4:], batch_size=1, shuffle=False, seed=seed)

    #train_y_augmented = train_y_datagen.flow(train_y, batch_size=1, shuffle=False, seed=seed)

    train_x1_augument_numpy = np.array([element for element in train_x1_augmented])
    #train_x2_augument_numpy = np.array([element for element in train_x2_augmented])
    #train_x3_augument_numpy = np.array([element for element in train_x3_augmented])
    #train_x4_augument_numpy = np.array([element for element in train_x4_augmented])
    #train_x5_augument_numpy = np.array([element for element in train_x5_augmented])

    train_x_augument_numpy = np.zeros((train_x1_augument_numpy.shape[0], train_x1_augument_numpy.shape[1], train_x1_augument_numpy.shape[2], 5))
    train_x_augument_numpy[...,0] = train_x1_augument_numpy
    #train_x_augument_numpy[...,1] = train_x2_augument_numpy
    #train_x_augument_numpy[...,2] = train_x3_augument_numpy
    #train_x_augument_numpy[...,3] = train_x4_augument_numpy
    #train_x_augument_numpy[...,4] = train_x5_augument_numpy
    #train_y_augument_numpy = np.array([element for element in train_y_augmented])
    print(train_x, train_x_augument_numpy)



if __name__ == "__main__":
    train_files, val_files, test_files = get_data_files(data="ca", label="Aorta")
    train_data, label_data = get_train_data_slices(train_files, tag ="Aorta")
    sitk_image = sitk.GetImageFromArray(label_data)
    sitk.WriteImage(sitk_image, "test_gt_Aorta.nii.gz")


    train_files, val_files, test_files = get_data_files(data="ca", label="RCA")
    train_data, label_data = get_train_data_slices(train_files, tag ="RCA")
    sitk_image = sitk.GetImageFromArray(label_data)
    sitk.WriteImage(sitk_image, "test_gt_RCA.nii.gz")

    train_files, val_files, test_files = get_data_files(data="ca", label="LM")
    train_data, label_data = get_train_data_slices(train_files, tag ="LM")
    sitk_image = sitk.GetImageFromArray(label_data)
    sitk.WriteImage(sitk_image, "test_gt_LM.nii.gz")

    #data_augumentation(train_data, label_data)
    print(label_data)
    print(label_data.shape)
    """plt.figure()
    plt.imshow(train_data[0][...,0], cmap="gray")
    plt.figure()
    plt.imshow(train_data[-1][...,0], cmap="gray")
    plt.figure()
    plt.imshow(train_data[label_data.shape[0]//2][...,0], cmap="gray")
    plt.show()"""
    #write_all_labels("../st.Olav/CT_FFR_3/CT_FFR_3_Segmentation/CT_FFR_3_Segmentation_0000/CT_FFR_3_Segmentation_0000_")
    #write_pridiction_to_file(train_data[...,2], label_data)

import os
import json
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# A path to a T1-weighted brain .nii image:
#t1_fn = '../Task02_Heart/imagesTr/la_003.nii'

# Read the .nii image containing the volume with SimpleITK:
#sitk_t1 = sitk.ReadImage(t1_fn)
#itk_label = sitk.ReadImage('../Task02_Heart/labelsTr/la_003.nii')
# and access the numpy array:
#t1 = sitk.GetArrayFromImage(sitk_t1)

def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):

    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    """out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]"""
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

    return resample.Execute(itk_image)

def scope_training_data(image, label):
    first_non_backgroud_slice = float('inf')
    last_non_backgroud_slice = -1
    # Normalizw images
    image -= np.min(image)
    image /= np.max(image)
    for i in range(image.shape[0]):
        if(1 in label[i]):
            if(i < first_non_backgroud_slice):
                first_non_backgroud_slice = i
            last_non_backgroud_slice = i
    return image[first_non_backgroud_slice:last_non_backgroud_slice+1], label[first_non_backgroud_slice:last_non_backgroud_slice+1]

def fetch_training_data_files():
    training_data_files = list()
    path = os.path.join(os.path.dirname(__file__), "Task08_HepaticVessel")
    count = 0
    with open("../Task08_HepaticVessel/dataset.json", "r+", encoding="utf-8") as f:
        data = json.load(f)
        for i in range(len(data["training"])):
            subject_files = list()
            subject_files.append(os.path.join(path.replace("/unet",""),data["training"][i]["image"].replace("./", "")))
            subject_files.append(os.path.join( path.replace("/unet",""),data["training"][i]["label"].replace("./", "")))
            training_data_files.append(tuple(subject_files))
            break
    return training_data_files

def get_preprossed_numpy_arrays_from_file(image_path, label_path):
    sitk_image  = sitk.ReadImage(image_path)
    sitk_label  = sitk.ReadImage(label_path)
    print(image_path)
    return scope_training_data(sitk.GetArrayFromImage(sitk_image), sitk.GetArrayFromImage(sitk_label))

#TODO fiks so indexes
def get_train_and_label_numpy(number_of_slices, train_list, label_list):
    train_data = np.zeros((number_of_slices, train_list[0].shape[1], train_list[0].shape[2]))
    label_data = np.zeros((number_of_slices, label_list[0].shape[1], label_list[0].shape[2]))
    index = 0
    for i in range(number_of_slices):
        for j in range(len(train_list)):
            for k in range(train_list[j].shape[0]):
                print(str(index+1) +"/" + str(number_of_slices))
                print("slices of image " + str(k+1) +"/" + str(train_list[j].shape[0]))
                print(train_list[j][k])
                train_data[index] = train_list[j][k]
                label_data[index] = label_list[j][k]
                index += 1
        break
    return train_data, label_data

def write_numpyimage_to_file(numpy_image, path):
    image = sitk.GetImageFromArray(numpy_image)
    sitk.WriteImage(path)

def read_numpyarray_from_file(path):
    image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image)

def show_nii_image(path, slice_nr):
    image = read_numpyarray_from_file(path)
    plt.figure()
    plt.imshow(image[slice_nr])


# Assume to have some sitk image (itk_image) and label (itk_label)
def get_training_data():
    traindata = []
    labeldata = []
    count_slices = 0
    files = fetch_training_data_files()
    for element in files:
        i, l = get_preprossed_numpy_arrays_from_file(element[0], element[1])
        count_slices += i.shape[0]
        traindata.append(i)
        labeldata.append(l)
    train_data, label_data = get_train_and_label_numpy(count_slices, traindata, labeldata)

    print("min: " + str(np.min(train_data)) +", max: " + str(np.max(train_data)))
    return train_data, label_data

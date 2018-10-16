import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# A path to a T1-weighted brain .nii image:
t1_fn = '../Task02_Heart/imagesTr/la_003.nii'

# Read the .nii image containing the volume with SimpleITK:
sitk_t1 = sitk.ReadImage(t1_fn)
itk_label = sitk.ReadImage('../Task02_Heart/labelsTr/la_003.nii')
# and access the numpy array:
t1 = sitk.GetArrayFromImage(sitk_t1)

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

# Assume to have some sitk image (itk_image) and label (itk_label)
def get_training_data():
    new_t1 = resample_img(sitk_t1, out_spacing=[4.0, 4.0, 4.0], is_label=False)
    print(sitk.GetArrayFromImage(new_t1).shape)
    print(sitk.GetArrayFromImage(new_t1))
    new_label_t1  = resample_img(itk_label, out_spacing=[2.0, 2.0, 2.0], is_label=True)
    print(sitk.GetArrayFromImage(new_label_t1).shape)
    traindata = sitk.GetArrayFromImage(new_t1)
    labeldata = sitk.GetArrayFromImage(new_label_t1)
    #plt.imshow(traindata[20])
    #plt.show()
    return traindata, labeldata

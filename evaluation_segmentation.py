# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:51:05 2020

@author: Kimberley Timmins


Evaluation for segmentation at ADAM challenge at MICCAI 2020
"""
from difflib import SequenceMatcher
import numpy as np
import os
import SimpleITK as sitk
from scipy import ndimage
import scipy.spatial
import evaluation_detection as ed


# Set the path to the source data (e.g. the training data for self-testing)
# and the output directory of that subject
#test_dir = r''  # For example: '/data/0'
#participant_dir = r''  # For example: '/output/teamname/0'


def do():
    """Main function"""
    result_filename = get_result_filename(participant_dir)
    test_image, result_image = get_images(os.path.join(test_dir, 'aneurysms.nii.gz'), result_filename)

    dsc = get_dsc(test_image, result_image)
    h95 = get_hausdorff(test_image, result_image)
    vs = get_vs(test_image, result_image)

    test_locations = ed.get_locations(os.path.join(test_dir, 'location.txt'))
    result_locations = get_center_of_mass(result_image)

    sensitivity, false_positives = ed.get_detection_metrics(test_locations, result_locations, test_image)
    
    print('Dice: %.3f (higher is better, max=1)' % dsc)
    print('HD: %.3f mm (lower is better, min=0)' % h95)
    print('VS: %.3f (higher is better, min=0)' % vs)
    print('Sensitivity: %.3f (higher is better, max=1)' % sensitivity)
    print('False Positive Count: %d (lower is better)' % false_positives)
    

def get_result_filename(dirname):
    """Find the filename of the result image.
    
    This should be result.nii.gz or result.nii. If these files are not present,
    it tries to find the closest filename."""

    files = os.listdir(dirname)
    
    if not files:
        raise Exception("No results in " + dirname)
    
    # Find the filename that is closest to either 'result.nii.gz' or 'result.nii'.
    ratios = [[SequenceMatcher(a=a, b=b).ratio() for b in ['result.nii.gz', 'result.nii']] for a in files]
    result_filename = files[int(np.argmax(np.max(ratios, axis=1)))]

    # Return the full path to the file.
    return os.path.join(dirname, result_filename)


def get_images(test_filename, result_filename):
    """Return the test and result images, thresholded and treated aneurysms removed."""
    test_image = sitk.ReadImage(test_filename)
    result_image = sitk.ReadImage(result_filename)
    
    assert test_image.GetSize() == result_image.GetSize()
    
    # Get meta data from the test-image, needed for some sitk methods that check this
    result_image.CopyInformation(test_image)
    
    # Remove treated aneurysms from the test and result images, since we do not evaluate on this
    treated_image = test_image != 2  # treated aneurysms == 2
    masked_result_image = sitk.Mask(result_image, treated_image)
    masked_test_image = sitk.Mask(test_image, treated_image)
    
    # Return two binary masks
    return masked_test_image > 0.5, masked_result_image > 0.5

        
def get_dsc(test_image, result_image):
    """Compute the Dice Similarity Coefficient."""
    test_array = sitk.GetArrayFromImage(test_image).flatten()
    result_array = sitk.GetArrayFromImage(result_image).flatten()
    
    test_sum = np.sum(test_array)
    result_sum = np.sum(result_array)
    
    if test_sum == 0 and result_sum == 0:
        # Perfect result in case of no aneurysm
        return np.nan
    elif test_sum == 0 and not result_sum == 0:
        # Some segmentations, while there is no aneurysm
        return 0
    else:
        # There is an aneurysm, return similarity = 1.0 - dissimilarity
        return 1.0 - scipy.spatial.distance.dice(test_array, result_array)
    

def get_hausdorff(test_image, result_image):
    """Compute the Hausdorff distance."""

    result_statistics = sitk.StatisticsImageFilter()
    result_statistics.Execute(result_image)
  
    if result_statistics.GetSum() == 0:
        hd = np.nan
        return hd

    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 3D
    e_test_image = sitk.BinaryErode(test_image, (1, 1, 1))
    e_result_image = sitk.BinaryErode(result_image, (1, 1, 1))

    h_test_image = sitk.Subtract(test_image, e_test_image)
    h_result_image = sitk.Subtract(result_image, e_result_image)

    h_test_indices = np.flip(np.argwhere(sitk.GetArrayFromImage(h_test_image))).tolist()
    h_result_indices = np.flip(np.argwhere(sitk.GetArrayFromImage(h_result_image))).tolist()

    test_coordinates = [test_image.TransformIndexToPhysicalPoint(x) for x in h_test_indices]
    result_coordinates = [test_image.TransformIndexToPhysicalPoint(x) for x in h_result_indices]
    
    def get_distances_from_a_to_b(a, b):
        kd_tree = scipy.spatial.KDTree(a, leafsize=100)
        return kd_tree.query(b, k=1, eps=0, p=2)[0]

    d_test_to_result = get_distances_from_a_to_b(test_coordinates, result_coordinates)
    d_result_to_test = get_distances_from_a_to_b(result_coordinates, test_coordinates)

    hd = max(np.percentile(d_test_to_result, 95), np.percentile(d_result_to_test, 95))
    
    return hd
    
       
def get_vs(test_image, result_image):
    """Volumetric Similarity.
    
    VS = 1 -abs(A-B)/(A+B)
    
    A = ground truth
    B = predicted     
    """
    
    test_statistics = sitk.StatisticsImageFilter()
    result_statistics = sitk.StatisticsImageFilter()
    
    test_statistics.Execute(test_image)
    result_statistics.Execute(result_image)
    
    numerator = abs(test_statistics.GetSum() - result_statistics.GetSum())
    denominator = test_statistics.GetSum() + result_statistics.GetSum()
    
    if denominator > 0:
        vs = 1 - float(numerator) / denominator
    else:
        vs = np.nan
            
    return vs
    

def get_center_of_mass(result_image):
    """Based on result segmentation, find coordinate of centre of mass of predicted aneurysms."""
    result_array = sitk.GetArrayFromImage(result_image)
    if np.sum(result_array) == 0:
        # no detections
        return np.ndarray((0, 3))

    structure = ndimage.generate_binary_structure(rank=result_array.ndim, connectivity=result_array.ndim)
   
    label_array = ndimage.label(result_array, structure)[0]
    index = np.unique(label_array)[1:]

    # Get locations in x, y, z order.
    locations = np.fliplr(ndimage.measurements.center_of_mass(result_array, label_array, index))
    return locations

  
if __name__ == "__main__":
    do()

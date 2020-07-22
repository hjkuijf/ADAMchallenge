# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:03:01 2020

@author: Kim


Test method for ADAM challenge 
"""

import SimpleITK as sitk
import numpy as np
import random
import os
from scipy import ndimage


def do():
	input_dir = '/input'
	output_dir = '/output'

	# Load the image
	tof_image = sitk.ReadImage(os.path.join(input_dir, 'pre', 'TOF.nii.gz'))

	# Binary threshold between 85% of maximum intensity and maxmium intensity
	intensity = sitk.MinimumMaximumImageFilter()
	intensity.Execute(tof_image)
	maximum = intensity.GetMaximum()
	thresh_image = sitk.BinaryThreshold(tof_image, lowerThreshold=maximum*0.80, upperThreshold=maximum)

	#dilate binary image to make lesions larger
	dilated_image = sitk.BinaryDilate(thresh_image, (2,2,2))
	 
	#connected components
	dilated_array = sitk.GetArrayFromImage(dilated_image)
	structure = ndimage.generate_binary_structure(rank=dilated_array.ndim, connectivity=dilated_array.ndim)
	label_array = ndimage.label(dilated_array, structure)[0]
	index = np.unique(label_array)[1:] 

	#take random  number of largest connected components
	#maximum number of detected aneurysms is 10, minimum is 1
	if  len(index) > 10:
		num = random.randint(1,10)
	else:
		num = len(index)  

	label_image = sitk.GetImageFromArray(label_array)
	result_image = sitk.BinaryThreshold(label_image, lowerThreshold = 1, upperThreshold = num)
	locations = np.fliplr(ndimage.measurements.center_of_mass(dilated_array, label_array, index[:num])).astype(int)
	np.savetxt(os.path.join(output_dir, 'result.txt'), locations, delimiter=',')
	print('Text file saved')
	sitk.WriteImage(result_image, os.path.join(output_dir, 'result.nii.gz'))
	print('Image saved')


if __name__ == "__main__":
    do()
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:51:05 2020

@author: Kimberley Timmins


Evaluation for segmentation at ADAM challenge at MICCAI 2020
"""
import difflib
import numpy as np
import os
import SimpleITK as sitk
from scipy import ndimage
import scipy.spatial
import evaluation_detect as ed

# Set the path to the source data (e.g. the training data for self-testing)
# and the output directory of that subject
testDir        = '' # For example: '/data/0'
participantDir = '' # For example: '/output/teamname/0'

def do():
    """Main function"""
    resultFilename = getResultFilename(participantDir)  
    testImage, resultImage = getImages(os.path.join(testDir, 'aneurysms.nii.gz'), resultFilename)
    
    dsc = getDSC(testImage, resultImage)
    h95 = getHausdorff(testImage, resultImage)
    vs = getVS(testImage, resultImage)
    
    testLocations   = ed.getLocations(os.path.join(testDir, 'location.txt')) 
    resultLocations = getCenterOfMassDetections(resultImage) 
    
    sensitivity, falsePositiveCount = ed.getDetectionMetrics(testLocations, resultLocations, testImage)     
    
    print('Dice',                                dsc,       '(higher is better, max=1)')
    print('HD',                                  h95, 'mm',  '(lower is better, min=0)')
    print('VS',                                   vs,       '(higher is better, min=0)')
    print('Sensitivity ',                sensitivity,       '(higher is better, max=1)')
    print('False Positive Count', falsePositiveCount,               '(lower is better)')
    

def getResultFilename(participantDir):
    """Find the filename of the result image.
    
    This should be result.nii.gz or result.nii. If these files are not present,
    it tries to find the closest filename."""
    files = os.listdir(participantDir)
    
    if not files:
        raise Exception("No results in "+ participantDir)
    
    resultFilename = None
    if 'result.nii.gz' in files:
        resultFilename = os.path.join(participantDir, 'result.nii.gz')
    elif 'result.nii' in files:
        resultFilename = os.path.join(participantDir, 'result.nii')
    else:
        
        maxRatio = -1
        for f in files:
            currentRatio = difflib.SequenceMatcher(a = f, b = 'result.nii.gz').ratio()
            if currentRatio > maxRatio:
                resultFilename = os.path.join(participantDir, f)
                maxRatio = currentRatio
                
    return resultFilename


def getImages(testFilename, resultFilename):
    """Return the test and result images, thresholded and treated aneurysms removed."""
    testImage   = sitk.ReadImage(testFilename)
    resultImage = sitk.ReadImage(resultFilename)
    
    assert testImage.GetSize() == resultImage.GetSize()
    
    # Get meta data from the test-image, needed for some sitk methods that check this
    resultImage.CopyInformation(testImage)
    
    # Remove treated aneurysms from the test and result images, since we do not evaluate on this
    treatedImage      = sitk.BinaryThreshold(testImage, 2, 3, 0, 1) # treated aneurysms == 2
    maskedResultImage = sitk.Mask(resultImage, treatedImage)
    maskedTestImage   = sitk.Mask(testImage, treatedImage)
    
    # Convert to binary mask
    if 'integer' in maskedResultImage.GetPixelIDTypeAsString():
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 1, 1000, 1, 0)
    else:
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 0.5, 1000, 1, 0)
        
    return maskedTestImage, bResultImage

        
def getDSC(testImage, resultImage):    
    """Compute the Dice Similarity Coefficient."""
    testArray   = sitk.GetArrayFromImage(testImage).flatten()
    resultArray = sitk.GetArrayFromImage(resultImage).flatten()
    
    testSum   = np.sum(testArray)
    resultSum = np.sum(resultArray)
    
    if testSum == 0 and resultSum == 0:
        # Perfect result in case of no aneurysm
        return None
    if testSum == 0 and not resultSum == 0:
        # Some segmentations, while there is no aneurysm
        return 0
    
    # There is an aneurysm, return similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray)   
    

def getHausdorff(testImage, resultImage):
    """Compute the Hausdorff distance."""
    
    resultStatistics = sitk.StatisticsImageFilter()
    resultStatistics.Execute(resultImage)
  
    if resultStatistics.GetSum() == 0:
        hd = None
        return hd
        
    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 3D
    eTestImage   = sitk.BinaryErode(testImage, (1,1,1) )
    eResultImage = sitk.BinaryErode(resultImage, (1,1,1) )
    
    hTestImage   = sitk.Subtract(testImage, eTestImage)
    hResultImage = sitk.Subtract(resultImage, eResultImage)    
    
    hTestArray   = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)   
        
    testCoordinates   = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hTestArray) ))]
    resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hResultArray) ))]    
    
    def getDistancesFromAtoB(a, b):    
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]
    
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)  
    
    hd = max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))
    
    return hd
    
       
def getVS(testImage, resultImage):   
    """Volumetric Similarity.
    
    VS = 1 -abs(A-B)/(A+B)
    
    A = ground truth
    B = predicted     
    """
    
    testStatistics   = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()
    
    testStatistics.Execute(testImage)
    resultStatistics.Execute(resultImage)
    
    numerator = abs(testStatistics.GetSum() - resultStatistics.GetSum())
    denominator = testStatistics.GetSum() + resultStatistics.GetSum() 
    
    if denominator > 0:
        vs = 1 - float(numerator) / denominator
    else:
        vs = None
            
    return vs
    

def getCenterOfMassDetections(resultImage):
    """Based on result segmentation, find coordinate of centre of mass of predicted aneurysms."""
    resultArray = sitk.GetArrayFromImage(resultImage)
    if np.sum(resultArray) == 0:
        #no detections
        return []
    structure = ndimage.generate_binary_structure(rank = resultArray.ndim, connectivity = resultArray.ndim)
   
    labelArray = ndimage.label(resultArray, structure)[0]
    index = np.unique(labelArray)[1:] 
    
    locations = ndimage.measurements.center_of_mass(resultArray, labelArray, index)    
    locationsFlipped = np.fliplr(locations) # put into x, y, z order
    return np.rint(locationsFlipped).astype("int").tolist() # round to nearest voxel

  
if __name__ == "__main__":
    do()    
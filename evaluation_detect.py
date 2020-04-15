# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:53:21 2020

@author: Kimberley Timmins

Evaluation of detection task at ADAM challenge MICCAI 2020
"""

import difflib
import numpy as np
import os
import SimpleITK as sitk

# Set the path to the source data (e.g. the training data for self-testing)
# and the output directory of that subject
testDir        = '' # For example: '/data/0'
participantDir = '' # For example: '/output/teamname/0'

def do():
    """Main function"""
    resultFilename = getResultFilename(participantDir)  
       
    testLocations = getLocations(os.path.join(testDir, 'location.txt'))
    resultLocations = getResult(resultFilename)
    testImage = sitk.ReadImage(os.path.join(testDir, 'aneurysms.nii.gz'))
    
    sensitivity, falsePositiveCount = getDetectionMetrics(testLocations, resultLocations, testImage)       

    print('Sensitivity ',                sensitivity, '(higher is better, max=1)')
    print('False Positive Count', falsePositiveCount,         '(lower is better)')
    

def getLocations(testFilename):
    """Return the locations and radius of actual aneurysms"""
    
    testLocations = []
    with open(testFilename, 'r') as f:
        for line in f:
            i = line.rstrip().split(', ')
            testLocations.append([int(x) for x in i[:3]] + [float(i[3])])
          
    for coord in testLocations:
        assert len(coord) == 4
    
    return testLocations


def getResultFilename(participantDir):
    """Find the filename of the result coordinate file.
    
    This should be result.txt  If this file is not present,
    it tries to find the closest filename."""
    
    files = os.listdir(participantDir)
    
    if not files:
        raise Exception("No results in "+ participantDir)
    
    resultFilename = None
    if 'result.txt' in files:
        resultFilename = os.path.join(participantDir, 'result.txt')
    else:
        # Find the filename that is closest to 'result.txt'
        maxRatio = -1
        for f in files:
            currentRatio = difflib.SequenceMatcher(a = f, b = 'result.txt').ratio()
            if currentRatio > maxRatio:
                resultFilename = os.path.join(participantDir, f)
                maxRatio = currentRatio
                
    return resultFilename
    
	
def getResult(resultFilename):
    """Read Result file and extract coordinates in list"""
    
    resultLocations = []
    with open(resultFilename, 'r') as f:
        for line in f:
            resultLocations.append([int(x) for x in line.rstrip().split(', ')])
    
    for coord in resultLocations:
        assert len(coord) == 3
        
    return resultLocations
   
   
def getDetectionMetrics(testLocations, resultLocations, testImage):
    """Calculate sensitivity and false positive count for each image.
    
    The distance between every result-locations and test-locations must be less
    than the radius."""
    
    testRadii = [coord[-1] for coord in testLocations]
    
    testCoords = [np.array(testImage.TransformIndexToPhysicalPoint(coord[:3])) for coord in testLocations]
    predCoords = [np.array(testImage.TransformIndexToPhysicalPoint(coord)) for coord in resultLocations]
    
    TP = 0
    for location, radius in zip(testCoords, testRadii):
        for detection in predCoords:
            distance = np.linalg.norm(detection - location)
            if distance <= radius:
                TP += 1
                break               
            
    FP = 0
    for detection in predCoords:
        found = False
        for location, radius in zip(testCoords, testRadii):
            distance = np.linalg.norm(location - detection)
            if distance <= radius:
                found = True 
        if not found:
            FP += 1
     
    if len(testLocations) == 0:
        sensitivity = None
    else:
        sensitivity = TP / len(testLocations)  
      
    return sensitivity, FP

  
if __name__ == "__main__":
    do()   
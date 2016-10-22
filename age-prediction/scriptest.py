# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:23:32 2016

@author: valentin
"""
#%% IMPORTATION oF THE NECESSARY LIBRARIES:
import numpy as np
import nibabel as nib
import imageAnalysis as ia
import importlib
import MachineLearningAlgorithms as ml

print("Libraries imported")

#%% RELOADING OF THE MODIFIED LIBRARIES:
ia = importlib.reload(ia)

print("Libraries reloaded")

#%% LOADING OF THE LABELS OF THE TRAINING DATASET:

# Loading of the labels of the training dataset:
label = np.genfromtxt('data/targets.csv', delimiter=',').astype(int)

# Number of labeled patients, i.e number of 3D MRI images:
nImages = label.size

# Min et max age of the patients:
ageMin = label.min()
ageMinIndex = label.argmin()
ageMax = label.max()
ageMaxIndex = label.argmax()

print("Labels Loaded")

#%% CREATION OF RANDOM DATASETS FOR TRAINING AND VALIDATION:

# Ratio of the overall training dataset used for real training and used for 
# validation:
ratioTV = .7

# Number of images used for training:
nTrain = round(ratioTV*nImages)

# Number of images used for validation:
nValid = nImages - nTrain

print(nTrain)
print(nValid)

shuffleIndex = np.arange(nImages)
np.random.shuffle(shuffleIndex)

# Indices of the training and validation datasets:
trainIndex = shuffleIndex[:nTrain-1]
validIndex = shuffleIndex[nTrain:]

# Labels of the training and validation datasets:
labelTrain =label[trainIndex]
labelValid =label[validIndex]

print ("Training and validation dataset indexes created. Average age:\n\
Training dataset: {} y.o\nvalidation dataset: {} y.o".format(int(round(labelTrain.mean())), \
int(round(labelValid.mean()))))

#%% CREATION OF THE TRAINING & VALIDATION DATASET:

# Creation of the variable imageDataSet containing all the data of the training
# dataset:
nImages = nTrain

# Size of the image:
xSize = 176
ySize = 208
zSize = 176

# Limit of the regions on which where is no relevant signal:
xLimMin = 14
xLimMax = 18
yLimMin = 12
yLimMax = 15
zLimMin = 3
zLimMax = 20

# Creation of a dictionary:
trainDataset = {}
validDataset = {}

# Loading of the images from the training dataset:
path = "data/" 
strDataset = "set_train"
dataSetIndex = trainIndex

for i, v in enumerate(dataSetIndex):
    #print(str(i) + " " + str(100.0*i/nTrain))
    imageName = "train_" + str(v+1)
    imagePath = path + "/" + strDataset + "/" + imageName + ".nii"

    imageRaw = nib.load(imagePath)

    # Tranforming the images into data (3d matrix):
    trainDataset[i] = imageRaw.get_data()[xLimMin:xSize-xLimMax, \
    yLimMin:ySize-yLimMax, zLimMin:zSize-zLimMax, 0]
    
dataSetIndex = validIndex

for i, v in enumerate(dataSetIndex):
    #print(str(i) + " " + str(100.0*i/nTrain))
    imageName = "train_" + str(v+1)
    imagePath = path + "/" + strDataset + "/" + imageName + ".nii"

    imageRaw = nib.load(imagePath)

    # Tranforming the images into data (3d matrix):
    validDataset[i] = imageRaw.get_data()[xLimMin:xSize-xLimMax, \
    yLimMin:ySize-yLimMax, zLimMin:zSize-zLimMax, 0]

# We create an object of the class ImageProperties from the imageData:
image_Young = ia.ImageProperties(trainDataset[0]**1.)
image_Old = ia.ImageProperties(trainDataset[1]**1.)

print("Training & validation dataset created")

#%% FEATURES EXTRACTION:

# We create an object of the class Features from the imageData:
dataSet = ml.Features(trainDataset)

""" We create a dictionary containing the features we want. It should respect 
the following construction rules:
    1) The key is the name of the function (string) that we want to use 
    to extract the features we wish. 
    2) The value represents the parameters of the used function knowing
    that:
        - the dataset should not be precised even if it is a parameter
        of the function.
        - the parameters can be either a dictionary: {"input1": value1,
        ..., "inputN": valueN} or a number. If it is a number it will 
        be interpreted as the polynomial order on which we want to fit 
        the given feature
"""
featureDic = {"gridOperation": { nGrid=(10,10), npoly=1, type2D="center",  \
                      axis=2, "mean"} } 

# Number of features:
#nFeatures = 


#%%
image_Young.compare(image_Old)



















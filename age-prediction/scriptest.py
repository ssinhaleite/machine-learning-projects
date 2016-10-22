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

print("\nLibraries imported")

#%% RELOADING OF THE MODIFIED LIBRARIES:
ia = importlib.reload(ia)
ml = importlib.reload(ml)
print("\nLibraries reloaded")

#%% LOADING OF THE LABELS OF THE TRAINING DATASET:

# Loading of the labels of the training dataset:
label = np.genfromtxt('data/targets.csv', delimiter=',').astype(int)

# Number of labeled patients, i.e number of 3D MRI images:
nSamples = label.size

# Min et max age of the patients:
ageMin = label.min()
ageMinIndex = label.argmin()
ageMax = label.max()
ageMaxIndex = label.argmax()
		
print("\nLabels loaded. There are " + str(nSamples) + " samples in the dataset")

#%% LOADING OF THE LABELED DATASET:
    
# Loading of the images from the training dataset:
path = "data/" 
strDataset = "set_train"

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

# Creation of the dictionary which will contain our labeled dataset:
datasetDic = {}

for i in range(nSamples):
    #print(str(i) + " " + str(100.0*i/nTrain))
    imageName = "train_" + str(i + 1)
    imagePath = path + "/" + strDataset + "/" + imageName + ".nii"

    imageRaw = nib.load(imagePath)

    # Tranforming the images into data (3d matrix):
    datasetDic[i] = imageRaw.get_data()[xLimMin:xSize-xLimMax, \
    yLimMin:ySize-yLimMax, zLimMin:zSize-zLimMax, 0]
    
    print("\nThe dataset dictionary containing all the 3D images of the labeled \
          dataset has been created")		

#%% FEATURES EXTRACTION:
ml = importlib.reload(ml)
# We create an object of the class Features from the trainDataset dictionary:
dataSet = ml.Features(datasetDic)

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
featureDic = {"gridOperation": { "nGrid":(50,50), "npoly":1, "type2D":"center",   \
                                 "axis":2, "typeOp":["mean"]} } 

# Use of the function featureExtraction to extract the features selected in the		
# dictionary featureDic:		
featureMatrix = dataSet.featureExtraction(**featureDic)

# Number of features:
nFeatures = featureMatrix.shape[1]		#nFeatures = 
print("The features matrix is computed. There are {} different features".format(nFeatures))		
#%% CREATION OF THE TRAINING AND VALIDATION DATASETS:		
# We create an object of the class Prediction from the trainDataset dictionary:		
data2Predict = ml.Prediction(featureMatrix, label)		
# The training dataset is randomaly divided into 2 datasets, one for the 		
# training, the over for the validation:		
# Ratio of the overall training dataset used for real training and used for 		
# validation:		
ratioTV = 1		
indexSplit, labelValidation = data2Predict.datasetSplit(ratioSplit = 0.8)		
print("Training and validation datasets created")		
#%% MODEL PARAMETER COMPUTATION:		
# Once we have the indices creating the 2 datasets, we can determine the two 		
# datasets using the function featureSplit:		
featureDic = data2Predict.featureSplit (**indexSplit)		
# The parameters of the model are computed on the training data set using the 		
# function modelParameters:		
parameters = data2Predict.modelParameters(featureDic, shrinkageParameter = 0, \
                                          technique = "LS")		
print("The parameters of the model has been computed")		
#%% VALIDATION OF THE MODEL:		
# We use the validation dataset to compute the mean squared error between the 		
# labels calculated with our model and the real labels:		
predictedData = data2Predict.predict(parameters, labelValidation)		
#%% LOADING OF THE NON-LABELED DATASET:
    
    
    

#%% PREDICTION FOR THE NON-LABELED DATASET:




#%%
image_Young.compare(image_Old)

#%% CREATION OF RANDOM DATASETS FOR TRAINING AND VALIDATION:

# Ratio of the overall training dataset used for real training and used for 
# validation:
ratioTV = 1

# Number of images used for training:
nTrain = round(ratioTV*nImages)

# Number of images used for validation:
nValid = nImages - nTrain

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


# We create an object of the class ImageProperties from the imageData:
image_Young = ia.ImageProperties(dataset[0]**1.)
image_Old = ia.ImageProperties(dataset[1]**1.)





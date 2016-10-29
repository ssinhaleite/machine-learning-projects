# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:23:32 2016

@author: valentin
@author: vleite
@author: kareny
"""
#%% IMPORTATION oF THE NECESSARY LIBRARIES:
import numpy as np
import nibabel as nib
import imageAnalysis as ia
import importlib
import MachineLearningAlgorithms as ml
import matplotlib.pyplot as plt

#valentin
#path = "D:/Machine Learning/"
#vleite
path = "./data/"

print("\nLibraries imported")

#%% RELOADING OF THE MODIFIED LIBRARIES:
ia = importlib.reload(ia)
ml = importlib.reload(ml)
print("\nLibraries reloaded")

#%% LOADING OF THE FUNCTIONS:
def loadData( path, strDataset, strName, nSamples ):
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
        imageName = strName + str(i + 1)
        imagePath = path + "/" + strDataset + "/" + imageName + ".nii"
        
        imageRaw = nib.load(imagePath)
        
        # Tranforming the images into data (3d matrix):
        datasetDic[i] = imageRaw.get_data()[xLimMin:xSize-xLimMax, \
        yLimMin:ySize-yLimMax, zLimMin:zSize-zLimMax, 0]
    
    return datasetDic

def featuresExtraction(datasetDic, featureDic):
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
#    featureDic = {"gridOperation": { "nGrid":(8,8,8), "npoly":1, "typeOp":["mean"]}, \
                  #"gridOperation": { "nGrid":(176,208), "npoly":1, "typeOp":["mean"]}, \
#                  "gridOperation": { "nGrid":(8,8,8), "npoly":2, "typeOp":["mean"]}, \
#                  "gridOperation": { "nGrid":(8,8,8), "npoly":2, "typeOp":["sum"]}, \
#                  "gridOperation": { "nGrid":(8,8,8), "npoly":1, "typeOp":["cov"]}, \
#                  "gridOperation": { "nGrid":(5,5), "npoly":1, "typeOp":["energy"]}, \
#                  "gridOperation": { "nGrid":(8,8), "npoly":1, "typeOp":["homogeneity"]}, \
#                  "gridOperation": { "nGrid":(8,8), "npoly":1, "typeOp":["dissimilarity"]} \
#                  "gridOperation": { "nGrid":(8,8), "npoly":1, "typeOp":["contrast"]} \
#                 } 

    # Use of the function featureExtraction to extract the features selected in the		
    # dictionary featureDic:		
    featureMatrix = dataSet.featureExtraction(**featureDic)

    # Number of features:
    nFeatures = featureMatrix.shape[1]
    print("The features matrix is computed. There are {} different features".format(nFeatures))
    
    return featureMatrix

print("\nFunctions loaded")

#%% LOADING OF THE LABELS OF THE TRAINING DATASET:
strLabel = "targets.csv"

# Loading of the labels of the training dataset:
label = np.genfromtxt(path+strLabel, delimiter=',').astype(int)

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
strDataset = "set_train"
strName = "train_"

datasetDic = loadData( path, strDataset, strName, nSamples )

print("\nThe dataset dictionary containing all the 3D images of the labeled dataset has been created")		

#%% FEATURES EXTRACTION:
ml = importlib.reload(ml)
featureDic = {}

## 2D grid operation:
#featureDic["gridOperation"] = { "nGrid":(15,15), "npoly":2, "axis":0, \  
#    "typeOp":["mean", "var"]}
#
## 3D grid operation:
featureDic["gridOperation"] = { "nGrid":(15,15,15), "npoly":1, "typeOp":["mean","var"]}

# 3D grid operation:
#featureDic["threshold"]  = { "nLevel":20, "thresholdType": "Energy", "axis":-1 }

#extracting features from the dataset
featureMatrix = featuresExtraction( datasetDic, featureDic)

#%% CROSS VALIDATION PREDICTION AND SCORES:
ml = importlib.reload(ml)
data2Predict = ml.Prediction(featureMatrix, label)    
    
MSECV = data2Predict.crossValidation(nFold=10, typeCV="random")

print("After cross-validation, we obtain a score of {}".format(MSECV))    

#%% COMPUTATION OF THE MODEL PARAMETERS ON THE WHOLE LABELED DATASET:
print(featureMatrix.shape)
_, modelParameters = data2Predict.buildClassifier(featureMatrix, \
             crossValid = False, labelTraining=label, classifier = "RidgeCV")

# Prediction of the data using the model parameters:
_, MSESelf = data2Predict.predict(modelParameters, featureMatrix,\
                              crossValid = False, labelValidation = label)  

print("Our model tested on the data used for training gives a score of {}".format(round((MSESelf),3)))      

#%% LOADING OF THE NON-LABELED DATASET:    
# Loading of the images from the test dataset:
strDataset = "set_test"
strName = "test_"

datasetTestDic = loadData( path, strDataset, strName, 138 )

print("\nThe dataset dictionary containing all the 3D images of the test dataset has been created")        
    
ml = importlib.reload(ml)

#extracting features from the dataset
featureMatrixTest = featuresExtraction(datasetTestDic, featureDic)

#%% PREDICTION FOR THE NON-LABELED DATASET:

# We create an object of the class Prediction from the trainDataset dictionary:        
unlabeledData= ml.Prediction(featureMatrixTest)

print(modelParameters.shape)
print(modelParameters.shape)
testPrediction = unlabeledData.predict(modelParameters, featureMatrixTest)

# Plot the predicted data and the true data:
plt.figure(102)

 # X-axis:
x = np.linspace(1, 138, 138)
    
# Plot of the predicted labels:
plt.plot(x, testPrediction, color="blue", linewidth=1, \
         linestyle='--', marker='o')
plt.xlabel("Patient number")
plt.ylabel("Age")


#%% WRITING ANSWER
fileStr = 'submission.csv'

fileIO = open( path + fileStr,'w' )
fileIO.write( 'ID,Prediction\n' )
answer = np.rint(testPrediction).astype(int)
for i in range( len( testPrediction ) ):
    fileIO.write( str(i+1) + ',' + str(answer[i]).strip('[]') + '\n' )
fileIO.close()


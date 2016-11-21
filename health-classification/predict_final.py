#%% IMPORTATION oF THE NECESSARY LIBRARIES:

import numpy as np
import nibabel as nib
import importlib

import sys
sys.path.append('../src')
import imageAnalysis as ia
import MachineLearningAlgorithms as ml
#import matplotlib.pyplot as plt

# Path where your training and test dataset are
path = "./data/"

print("\nLibraries imported")


#%% RELOADING OF THE MODIFIED LIBRARIES:
    
ia = importlib.reload(ia)
ml = importlib.reload(ml)

print("\nLibraries reloaded")


#%% LOADING OF THE FUNCTIONS USED IN THE SCRIPT:
def loadData( path, strDataset, strName, nSamples ):
    """ To load the 3D MRI images of a dataset
    
    INPUTS:
        path -- string, path of your repository
        strDataset -- string, name of the dataset folder
        strName -- string, common name of each file 
        nSamples -- integer, number of elements you want to pick up in the 
        dataset folder  
    OUTPUTS:
        datasetDic -- dictionary, the index of the dictionary corresponds to
        the file index of the 3D MRI images. And the value of the dictionary
        is a 3D np.array cropped such that we do not take into account the 
        parts of the 3D images containing only zeros. 
    """
    # Size of the image:
    xSize = 176
    ySize = 208
    zSize = 176

    # Limits of the regions of interest of the data:
    xLimMin = 14
    xLimMax = 18
    yLimMin = 12
    yLimMax = 15
    zLimMin = 3
    zLimMax = 20

    # Creation of the dictionary which will contain our dataset:
    datasetDic = {}

    for i in range(nSamples):
        # Complete path of the i-th file of the dataset:
        imageName = strName + str(i + 1)
        imagePath = path + "/" + strDataset + "/" + imageName + ".nii"
        
        # Loading of the 3D images using a function from the nibabel library
        imageRaw = nib.load(imagePath)
        
        # Tranforming the images into data (3d np.array):
        datasetDic[i] = imageRaw.get_data()[xLimMin:xSize-xLimMax, \
        yLimMin:ySize-yLimMax, zLimMin:zSize-zLimMax, 0]
    
    return datasetDic


def featuresExtraction(datasetDic, featureDic):
    """ To extract the relevant features from the input data
    
    INPUTS:
        datasetDic -- dictionary, it contains all the 3D MRI images from the 
        dataset
        featureDic -- dictionary, the key of the dictionary is the name of the 
        function (string) that we want to use to extract the features we wish. 
        The value contains the inputs variables of the functions used to 
        compute the features. it can can be either a dictionary: 
        {"input1": value1,..., "inputN": valueN} or a number. If it is a number 
        it will be interpreted as the polynomial order on which we want to fit 
        the given features  
    OUTPUTS:
        featureMatrix -- dictionary, the index of the dictionary corresponds to
        the file index of the 3D MRI images. And the value of the dictionary
        is a 1D np.array containing all the computed features. 
        
    """
    
     # We create an object of the class Features from the dataset dictionary:
    dataSet = ml.Features(datasetDic)
    
    # Use of the function featureExtraction (from the MachineLearningAlgorithms
    # module) to extract the features selected in the dictionary featureDic:        
    featureMatrix = dataSet.featureExtraction(**featureDic)

    # Number of features:
    nFeatures = featureMatrix.shape[1]
    print("\nThe features matrix is computed. There are {} different features".format(nFeatures))
    
    return featureMatrix

print("\nFunctions loaded")


#%% LOADING OF THE LABELS OF THE TRAINING DATASET:

# File name where is the labels of the training dataset:
strLabel = "targets.csv"

# Loading of the labels of the training dataset:
label = np.genfromtxt(path+strLabel, delimiter=',').astype(int)

# Number of labeled patients, i.e number of 3D MRI images:
nSamples = label.size
    
print("\nLabels loaded. There are " + str(nSamples) + " samples in the dataset")


#%% LOADING OF THE LABELED DATASET:

# Name of the training dataset folder
strDataset = "set_train"

# Common name of each file of the training dataset folder
strName = "train_"

# Loading of the images from the training dataset and saving in a dictionary:
datasetDic = loadData( path, strDataset, strName, nSamples )

print("\nThe dataset dictionary containing all the 3D images of the labeled \
dataset has been created")        


#%% FEATURES EXTRACTION:
    
ml = importlib.reload(ml)

# We create a dictionary containing the features we want. It should respect 
# the following construction rules:
# 1) The key is the name of the function (string) that we want to use 
# to extract the features we wish. 
# 2) The value represents the parameters of the used function knowing
# that:
#     - the parameters can be either a dictionary: {"input1": value1,
#     ..., "inputN": valueN} or a number. If it is a number it will 
#     be interpreted as the polynomial order on which we want to fit 
#     the given feature
featureDic = {}

# Feature dictionary used for obtaining our best score:
featureDic["gridOperation"] = { "nGrid":(9,9,9), "npoly":1, \
                                "typeOp":["mean","var"]}
            
# Extraction of the features we want from the training dataset:
featureMatrix = featuresExtraction( datasetDic, featureDic)


#%% CROSS VALIDATION PREDICTION AND SCORES:
    
ml = importlib.reload(ml)

# We create an object of the class Prediction to be able to use the functions
# of this class in particular for predicting the data:
data2Predict = ml.Prediction(featureMatrix, label)    

#Build n-classifiers and get its answers and then, vote.
#MSECV, classifiers = data2Predict.crossValidation(nFold=10, typeCV="random")
#print("After cross-validation, we obtain a score of {}".format(MSECV))    


#%% COMPUTATION OF THE MODEL PARAMETERS ON THE WHOLE LABELED DATASET:
    
# After having checked the accuracy of our feature selection and linear 
# regression method, the parameters of our model are determined over the 
# whole training dataset: 

#clf = data2Predict.buildClassifier(featureMatrix, \
#             label, method = "RandomForestClassifier" )

classifiersArray = []

classifiersType = []
classifiersType.append("RandomForestClassifier")
classifiersType.append("SVC")
classifiersType.append("GaussianNB")
classifiersType.append("GaussianNB_isotonic")
classifiersType.append("GaussianNB_sigmoid")
classifiersType.append("MLPClassifier")
classifiersType.append("KNeighborsClassifier")
classifiersType.append("GaussianProcess")
classifiersType.append("AdaBoostClassifier")
classifiersType.append("VotingClassifier")

i = 0
for classifier in classifiersType:
    classifiersArray.append( data2Predict.buildClassifier(featureMatrix, \
             label, method = classifier) )
    i += 1

# Prediction of the data using the model parameters:
for i in range(len(classifiersArray)):
    _, MSESelf = data2Predict.predict(0, featureMatrix,\
                              classifier=classifiersArray[i], labelValidation = label)
    print("The model {} tested on the data used for training gives a score of {}".format( classifiersType[i], round(MSESelf,3)))      


#%% LOADING OF THE NON-LABELED DATASET:    
    
ml = importlib.reload(ml)

# Name of the test dataset folder
strDataset = "set_test"

# Common name of each file of the test dataset folder
strName = "test_"

# Loading of the images from the test dataset and saving in a dictionary:
datasetTestDic = loadData( path, strDataset, strName, 138 )

print("\nThe dataset dictionary containing all the 3D images of the test \
      dataset has been created")        

# Extraction of the features of the test dataset:
featureMatrixTest = featuresExtraction(datasetTestDic, featureDic)


#%% PREDICTION FOR THE NON-LABELED DATASET:

ml = importlib.reload(ml)
    
# We create an object of the class Prediction from the test dataset dictionary:        
unlabeledData= ml.Prediction(featureMatrixTest)

# The labels of the test data set are predicted using the parameters of our 
#model:
testPredictionArray = []
for i in range(len(classifiersArray)):
    testPredictionArray.append(unlabeledData.predict(0, featureMatrixTest, classifier=classifiersArray[i]))

## Plot the predicted data and the true data:
#plt.figure(102)
#
# # X-axis:
#x = np.linspace(1, 138, 138)
#    
## Plot of the predicted labels:
#plt.plot(x, testPrediction, color="blue", linewidth=1, \
#         linestyle='--', marker='o')
#plt.xlabel("Patient number")
#plt.ylabel("Age")

print("\n The prediction for the non-labeled dataset")    

#%% WRITING OF THE PREDICTION INTO A .CSV FILE:

# Name of the file which will contain our predictions for the test dataset:
for i in range(len(classifiersArray)):
    fileStr = classifiersType[i]+ ".csv"

    fileIO = open( path + fileStr,'w' )
    fileIO.write( 'ID,Prediction\n' )
    answer = testPredictionArray[i][:,1]
    for i in range( len( answer ) ):
        fileIO.write( str(i+1) + ',' + str(answer[i]).strip('[]') + '\n' )
    fileIO.close()

print("\n The prediction has been written in a .csv file")    


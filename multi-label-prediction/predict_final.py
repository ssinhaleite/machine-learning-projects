#%% IMPORTATION oF THE NECESSARY LIBRARIES:

CV=True
EnsembleSelection = False

import numpy as np
import nibabel as nib
import importlib
import time

import sys
sys.path.append('../src')
import imageAnalysis as ia
import MachineLearningAlgorithms as ml
#import matplotlib.pyplot as plt

# Path where your training and test dataset are
path = "./data/"

print("\nLibraries imported")


#% RELOADING OF THE MODIFIED LIBRARIES:
    
ia = importlib.reload(ia)
ml = importlib.reload(ml)

print("\nLibraries reloaded")


#% LOADING OF THE FUNCTIONS USED IN THE SCRIPT:
def loadData( path, strDataset, strName, nSamples, begin = 0 ):
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
    xLimMin = 20
    xLimMax = 20
    yLimMin = 20
    yLimMax = 20
    zLimMin = 10
    zLimMax = 20

    # Creation of the dictionary which will contain our dataset:
    datasetDic = {}

    for i in range(begin, nSamples):
        # Complete path of the i-th file of the dataset:
        imageName = strName + str(i + 1)
        imagePath = path + strDataset + "/" + imageName + ".nii"
        
        # Loading of the 3D images using a function from the nibabel library
        imageRaw = nib.load(imagePath)
        
        # Tranforming the images into data (3d np.array):
        datasetDic[i] = imageRaw.get_data()[xLimMin:xSize-xLimMax, \
        yLimMin:ySize-yLimMax, zLimMin:zSize-zLimMax, 0]
    
    return datasetDic

def preprocessing(datasetDic):    
    """ To transform the images such that their means are equal to 0 and their
        standard deviation are 1.    
    """
    # To each voxels of a given image, the image mean value is subtracted:
    for n, image in datasetDic.items():
        # Mean value of the image:
        meanImage = np.mean(image)
        # Standard deviation of the image:
        standardDeviation = np.std(image)
        
        # The 
        datasetDic[n] = (image - meanImage) / standardDeviation

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
    featureMatrix, binEdges = dataSet.featureExtraction(**featureDic)

    # Number of features:
    nFeatures = featureMatrix.shape[1]
    print("\nThe features matrix is computed. There are {} different features".format(nFeatures))
    
    return featureMatrix, binEdges

print("\nFunctions loaded")


#%% LOADING OF THE LABELS OF THE TRAINING DATASET:

# File name where is the labels of the training dataset:
strLabel = "targets.csv"

# Loading of the labels of the training dataset:
label = np.genfromtxt(path+strLabel, delimiter=',').astype(int)

# Number of labeled patients, i.e number of 3D MRI images:
nSamples = int(label.size / 3.)

# Creation of the array containing the labels for gender, age & health status:
labelGender = np.zeros(nSamples)
labelAge = np.zeros(nSamples)
labelHealth = np.zeros(nSamples)

# Creation of the 8 classes:
# Male, Old, Sick - MOS
# Male, Old, Health - MOH
# Male, Young, Sick - MYS
# Male, Young, Health - MYH
# Female, Old, Sick - FOS
# Female, Old, Health - FOH
# Female, Young, Sick - FYS
# Female, Young, Health - FYH
nClasses = 8

classDic ={}
for i in range(nClasses):
    classDic[i] = []

# Creation of the array containing the classes for each patient:
classPatient = np.zeros(nSamples)
    
for i in range(nSamples):
    label_i = label[i]
    labelGender[i] = label_i[0]
    labelAge[i] = label_i[1]
    labelHealth[i] = label_i[2]

    # Determination of the class of the patient:
    classID = 4*labelGender[i] + 2*labelAge[i] + labelHealth[i] 
    classPatient[i] = classID 
    classDic[classID].append(i)
    
# Class(es) with no label:
nElementClass = np.zeros(nClasses)
emptyClass = []
for i in range(nClasses):
    nElementClass[i] = len(classDic[i])
    if nElementClass[i] == 0:
       emptyClass.append(i)
nonEmptyClasses = np.delete(np.arange(nClasses),emptyClass)
nFinalClasses = len(nonEmptyClasses)
   
# Porportion of each label:
FemaleProportion = round(100 * np.sum(labelGender) / float(nSamples), 2)
youngProportion = round(100 * np.sum(labelAge) / float(nSamples), 2)
healthyProportion = round(100 * np.sum(labelHealth) / float(nSamples), 2)
classProportion = np.round(100 * nElementClass / float(nSamples), 2)
    
print("\nLabels loaded. There are {} samples in the dataset.\n 1) Female({}%) - Male({}%) \n 2) Young({}%) - Old({}%) \n 3) Healthy({}%) - Sick({}%) \n".format(nSamples, \
        FemaleProportion, 100-FemaleProportion, \
        youngProportion, 100-youngProportion, \
        healthyProportion, 100-healthyProportion ))
print("--> MOS({}%) - MOH({}%) - MYS({}%) - MYH({}%) - FOS({}%) - FOH({}%) - FYS({}%) - FYH({}%)" \
      .format( classProportion[0], classProportion[1], classProportion[2], classProportion[3], \
      classProportion[4], classProportion[5], classProportion[6], classProportion[7]))

# We keep only non-empty classes:
classLabel = np.zeros([nFinalClasses, 3])
for i in range(nFinalClasses):
    binaryClass = bin(nonEmptyClasses[i])[2:].zfill(3)
    for k in range(3):
        classLabel[i,k] = int(binaryClass[k])
        
# Matrix containing the type of labels necessary to use multiclass sklearn classifier:
#labelType = np.zeros(nFinalClasses, nFinalClasses)
multiLabels = np.zeros([nSamples,3])
label = np.zeros(nSamples)
for i in range(nSamples):
    label[i] = np.argmin(np.abs(nonEmptyClasses - classPatient[i]))
    multiLabels[i,:] = classLabel[int(label[i]),:]
label = label.astype(int)  

#%% LOADING OF THE LABELED DATASET:

# Name of the training dataset folder
strDataset = "set_train"

# Common name of each file of the training dataset folder
strName = "train_"

# Loading of the images from the training dataset and saving in a dictionary:
datasetDic = loadData( path, strDataset, strName, nSamples )

#% Preprocessing: For each image, the mean of the image is subtracted:
datasetDic = preprocessing(datasetDic)
#imageCroped = ia.ImageProperties(datasetDic[100])
#imageCroped.toPlot()
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
                                "typeOp":["mean", "median", "variance"], "binEdges":45}

# Extraction of the features we want from the training dataset:
featureMatrix, binEdges = featuresExtraction( datasetDic, featureDic)

#%% METHOD SELECTION AND INPUTS
# Chosen method used for classification or regression --> methodML
# Dictionary containing the input of the chosen method --> methodDic                    

methodML = "AdaBoost"
adaBoostDic={"n_estimators":150, "learning_rate":0.005}

methodML = "Bagging"
baggingDic={"n_estimators":100, "n_jobs":-1, "bootstrap_features": False}

methodML = "Gradient Boosting"
gradBoostDic={"n_estimators":100, "learning_rate":0.05, "max_depth":1, "loss": "deviance"}

methodML = "Random Forest"
rdmForestDic={"n_estimators":1000, "criterion":"gini", "class_weight":None, \
                     "bootstrap":True, "oob_score":True, "n_jobs":-1}

methodML = "SVM"
svmDic= {"C": 1.0, "kernel":"poly", "degree":1, "probability":True}

methodML = "SVC"
svcDic= {"C": 1.0, "cache_size" : 200, "decision_function_shape":"ovr", "kernel":"rbf", \
         "probability":True, "shrinking":True, "verbose":False }

methodDic =[svmDic, rdmForestDic, adaBoostDic, baggingDic, gradBoostDic]
methodML = ["SVM", "Random Forest", "AdaBoost", "Bagging", "Gradient Boosting"]

methodDic =[rdmForestDic]
methodML = ["Random Forest"]

methodDic =[svmDic]
methodML = ["SVM"]

methodDic =[baggingDic]
methodML = ["Bagging"]

methodDic =[gradBoostDic]
methodML = ["Gradient Boosting"]

methodDic =[adaBoostDic]
methodML = ["AdaBoost"]

methodDic =[rdmForestDic]
methodML = ["Multi Output RF"]

methodDic =[svmDic]
methodML = ["Multi SVM"]

#methodDic =[rdmForestDic]
#methodML = ["Multi Random Forest"]

#methodDic =[svmDic, rdmForestDic]
#methodML = ["Multi SVM", "Multi Random Forest"]

# balanced_subsample / gini / entropy
# Type of kernel:
# kernel -->"linear" / "poly" / "rbf" / "sigmoid"
#

# Number of methods used:
nMethods = len(methodML)

if nMethods > 1:
    print("There are {} methods to train our model: ".format(nMethods), end="")
else:
    print("The method to train our model is ", end="")

for i in range(nMethods):
    if i < len(methodML)-2:
        print(methodML[i] + ", ", end="")
    elif i == len(methodML)-2:
        print(methodML[i] + " and ", end="")
    else:
        print(methodML[i])

#%% CROSS VALIDATION PREDICTION AND SCORES:
    
multiClass = True

# We create an object of the class Prediction to be able to use the functions
# of this class in particular for predicting the data:
ml = importlib.reload(ml)

if multiClass == False:
    multiLabels=[]
    classToLabels=[]

data2Predict = ml.Prediction(featureMatrix, label, multiLabels=multiLabels, \
                 classToLabels=classLabel)

if CV:
    ml = importlib.reload(ml)
    
    # We use cross validation to check the variance of the model
    if nMethods == 1:
        score, varianceModel = data2Predict.crossValidation(methodDic, nFold=10,  \
                                  typeCV="random", methodList=methodML, \
                                  stepSize = 0.01)
    else:
        score, weightModel = data2Predict.crossValidation(methodDic, nFold=10,  \
                                  typeCV="random", methodList=methodML, \
                                  stepSize = 0.001)

    print("After cross-validation, we obtain a score of {} with a variance of {}".format(score[0], varianceModel))

#%% ENSEMBLE SELECTION:
if EnsembleSelection:
    ml = importlib.reload(ml)
    ensembleSelectionChosen = True
    
    # We create an object of the class Prediction to be able to use the functions
    # of this class in particular for predicting the data:
    data2Predict = ml.Prediction(featureMatrix, label)  
    
#    methodDic =[svmDic, rdmForrestDic, adaBoostDic, baggingDic, gradBoostDic]
#    methodML = ["SVM", "Random Forest", "AdaBoost", "Bagging", "Gradient Boosting"]
    
    classifierList, weightModel, score = data2Predict.ensembleSelection(methodDic,\
                                  Ratio=0.7, typeDataset="random", methodList=methodML,\
                                  stepSize=0.001)

#%%
ensembleSelectionChosen = False
weightModel = np.ones([nMethods])/nMethods
#methodDic =[svmDic, rdmForrestDic, adaBoostDic, baggingDic, gradBoostDic]
#methodML = ["SVM", "Random Forest", "AdaBoost", "Bagging", "Gradient Boosting"]
#% COMPUTATION OF THE MODEL PARAMETERS ON THE WHOLE LABELED DATASET:

if ensembleSelectionChosen is False:
    # After having checked the accuracy of our feature selection and linear 
    # regression method, the parameters of our model are determined over the 
    # whole training dataset:
    classifierList=[]
    error=[]   

    for i in range(nMethods): 
        classifierList.append(data2Predict.buildClassifier(featureMatrix, label, methodDic[i], method=methodML[i]))

        # Prediction of the data using the model parameters:
        _, error_i = data2Predict.predict(featureMatrix, method=methodML[i], \
                                      labelValidation = label, classifier=classifierList[i])
        error.append(error_i)

    modelError = np.mean(weightModel*error)
    print(nMethods)
    print("Our model tested on the data\
     used for training gives a score of {}".format(round(modelError,3)))      

#%% LOADING OF THE NON-LABELED DATASET:
ml = importlib.reload(ml)

# Name of the test dataset folder
strDataset = "set_test"

# Common name of each file of the test dataset folder
strName = "test_"

# Loading of the images from the test dataset and saving in a dictionary:
datasetTestDic = loadData( path, strDataset, strName, 138 )
#Preprocessing: For each image, the mean of the image is subtracted:
datasetTestDic = preprocessing(datasetTestDic)
nSampleTest = len(datasetTestDic)

print("\nThe dataset dictionary containing all the 3D images of the test \
      dataset has been created")

# Feature dictionary used for obtaining our best score:
for k,v in featureDic.items():
    if ("subhistogram" in featureDic[k]["typeOp"]) or ("histogram" in featureDic[k]["typeOp"]):
        featureDic[k]["binEdges"] = binEdges

# Extraction of the features of the test dataset:
featureMatrixTest, _ = featuresExtraction(datasetTestDic, featureDic)


#%% PREDICTION FOR THE NON-LABELED DATASET:
ml = importlib.reload(ml)
    
# We create an object of the class Prediction from the test dataset dictionary:        
unlabeledData = ml.Prediction(featureMatrixTest, classToLabels=classLabel)

# The labels of the test data set are predicted using the parameters of our 
#model:
testPrediction = np.zeros([nSampleTest])

for i in range(nMethods):
    predictionModel = unlabeledData.predict(featureMatrixTest, method=methodML[i], \
                                  labelValidation = [], classifier=classifierList[i])
    testPrediction += weightModel[i]*predictionModel
    
testPrediction = testPrediction.astype(int)

# Predicted labels (gender, age, health):
predictedLabels = -np.ones([nSampleTest,3])
for i in range(nSampleTest):
    predictedLabels[i] = classLabel[testPrediction[i],:]
    
print(predictedLabels)
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

date = (time.strftime("%d-%m-%Y %Hh%Mm%S"))

methods = ""

# Name of the file which will contain our predictions for the test dataset:
for i in range(nMethods):
    methods += methodML[i] + " "

fileStr = methods + date + ".csv"
#fileStr = "final_sub.csv"
fileIO = open( path + fileStr,'w' )
fileIO.write( 'ID,Sample,Label,Predicted\n' )
answer = testPrediction
for i in range( 3*nSampleTest ):
    ID = str(i)
    
    # Sample number:
    sample = str(i // 3)
    
    # Type of prediction:
    if i % 3 == 0:
        predictionType = "gender"
    elif i % 3 == 1:
        predictionType = "age"
    elif i % 3 == 2:
        predictionType = "health"

    # Predicted data:
    labelstr = str(bool(predictedLabels[int(sample), i%3]))
    
    fileIO.write( ID + ',' + sample + ',' + predictionType + "," + labelstr + '\n' )
fileIO.close()

print("\n The prediction has been written in a .csv file")


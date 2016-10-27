# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:06:58 2016

@author: valentin
"""

import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
from sklearn import preprocessing
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.naive_bayes import GaussianNB

#for image features
from skimage.feature import greycomatrix, greycoprops
from skimage import img_as_ubyte

class Features:
    
    def __init__(self, datasetDic):
        self.dataset = datasetDic
        self.nData =  len( self.dataset ) 
        
        self.size3D = self.dataset[0].shape
        (self.sizeX, self.sizeY, self.sizeZ) = self.size3D
        
        # Central positions along x, y, z:
        self.X0 = int(float(self.sizeX)/2)
        self.Y0 = int(float(self.sizeY)/2)
        self.Z0 = int(float(self.sizeZ)/2)
        
        
    def featureExtraction(self, **featureType):
        """
        DEFINITION:
        This function computes the feature matrix for a dataseet. This can be 
        used either to the determine the parameter matrix of our model if we
        have labeled data (= training dataset) or to realize prediction (= 
        validation dataset or test dataset).
        
        INPUT VARIABLES:
        featureType: dictionary. It should respect the following construction 
        rules:
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
        # Initialisation of the featureMatrix which will contain the values  \
        # of all the features computed for each element of the dataset:
        featureMatrix = np.ones([self.nData,1]) # This corresponds to the bias
        
        # Database dictionary containing the functions potentially used for 
        # features extraction:
        functionDic = {} # should be fulfill as soon as we add a new function
                         # for feature extraction !!!
        functionDic["gridOperation"] = self.gridOperation
                    
        # In the following loop, we call the functions allowing to compute the \
        # features chosen by the users in the dictionary featureType:
        for k,v in featureType.items():
            if isinstance(v,dict): 
                localFeatures = functionDic[k](**v)
            else:
                if v < 1:
                    v = 1
                localFeatures = functionDic[k](npoly=v)	
                
            featureMatrix = np.concatenate((featureMatrix, localFeatures), axis=1)
            
            return featureMatrix
            
###############################################################################          
            
    def gridOperation(self, typeOp=["mean"], nGrid=(10,10), npoly=1, type2D="center", axis=2):
   
        # Dictionary containing the dataset:
        datasetDic = self.dataset
        
        if npoly < 1:
            npoly = 1
            
        if len(nGrid) < 3:
            nDimension = 2
        else:
            nDimension = 3
            
        # Number of operations required by the user:
        nOp = len(typeOp)
        
        # Length (in points) of the grid along the different dimension:     
        xlength = int(round(self.sizeX*nGrid[0]/100))
        ylength = int(round(self.sizeY*nGrid[1]/100))
                    
        if nDimension == 3:
            zlength = int(round(self.sizeZ*nGrid[2]/100))
        
        # Number of subdivisions of the grid along each dimension: 
        nGridX = int(np.ceil(self.sizeX / xlength))
        nGridY = int(np.ceil(self.sizeY / ylength))
        if nDimension == 3:
            nGridZ = int(np.ceil(self.sizeZ / zlength))

        # Creation of the featureMatrix containing the features in a 4D/5D matrix:
        if nDimension == 2:
            featureMatrix = np.empty([nGridX, nGridY, nOp, npoly])
            features = np.empty([self.nData, nGridX*nGridY*nOp*npoly])
        elif nDimension == 3:
            featureMatrix = np.empty([nGridX, nGridY, nGridZ, nOp, npoly])
            features = np.empty([self.nData, nGridX*nGridY*nGridZ*nOp*npoly])

        if nGrid[0]==176:
            print(xlength)
            print(ylength)
            print(zlength)
            print(nGridX)
            print(nGridY)
            print(nGridZ)
            print(features.shape)

            
        for iDataset in range(self.nData):
            # Case of a 2D grid:
            if nDimension == 2:
                # The grid operation is repeated for the 2D image given by the
                # input variable type2D:
                #   type2D == "center" --> the following 2D image:
                #       Image3D [:,:,z0] if axis == 2 with z0 = zlength / 2
                #       Image3D [:,y0,:] if axis == 1 with y0 = ylength / 2
                #   type2D == "sum" --> the following 2D image:
                #       sum(Image3D [:,:,:]) along the z-dimension if axis == 2 
                #       sum(Image3D [:,:,:]) along the y-dimension if axis == 1
                #   type2D == n (n is an integer) --> the following 2D image:
                #       Image3D [:,:,5] if axis == 2 
                #       Image3D [:,5,:] if axis == 1"""                  
                if type2D == "center":
                    if axis == 0:
                        image2D = datasetDic[iDataset][self.X0,:,:]
                    elif axis == 1:
                        image2D = datasetDic[iDataset][:,self.Y0,:]
                    elif axis == 2:
                        image2D = datasetDic[iDataset][:,:,self.Z0]
                elif type2D == "sum":
                    image2D = datasetDic[iDataset].sum(axis)
                else:
                    if axis == 0:
                        image2D = datasetDic[iDataset][type2D,:,:]
                    elif axis == 1:
                        image2D = datasetDic[iDataset][:,type2D,:]
                    elif axis == 2:
                        image2D = datasetDic[iDataset][:,:,type2D]
                
            for iX in range(nGridX):
                for iY in range(nGridY):
                    if nDimension == 2:
                        gridZone = image2D[iX*xlength : (iX+1)*xlength, \
                                           iY*ylength : (iY+1)*ylength]      
                        for iPolyOrder in range(npoly):
                            for iOp, Op in enumerate(typeOp):
                                if Op in ["average", "Average", "mean"]:
                                    featureMatrix[iX, iY, iOp, iPolyOrder] = \
                                    np.mean(gridZone)**(iPolyOrder+1)
                                elif Op in ["max", "Max"]:
                                    featureMatrix[iX, iY, iOp, iPolyOrder] = \
                                    np.amax(gridZone)**(iPolyOrder+1)
                                elif Op in ["min", "Min"]:
                                    featureMatrix[iX, iY, iOp, iPolyOrder] = \
                                    np.amin(gridZone)**(iPolyOrder+1)
                                elif Op in ["variance", "Var", "Expectation", \
                                            "expectation"]:
                                    featureMatrix[iX, iY, iOp, iPolyOrder] = \
                                    np.var(gridZone)**(iPolyOrder+1)
                                elif Op in ["covariance", "cov"]:
                                    featureMatrix[iX, iY, iOp, iPolyOrder] = \
                                    np.cov(gridZone)**(iPolyOrder+1)
                                elif Op in ["sum", "Sum"]:
                                    featureMatrix[iX, iY, iOp, iPolyOrder] = \
                                    np.sum(gridZone)**(iPolyOrder+1)
                                elif Op in ["energy"]:
                                    #calcula a GLCM
                                    gridZone = img_as_ubyte(gridZone)
                                    glcm = greycomatrix(gridZone, [1], [0], symmetric=False, normed=True)
                                    featureMatrix[iX, iY, iOp, iPolyOrder] = \
                                    greycoprops(glcm, 'energy')[0, 0]
                                elif Op in ["contrast"]:
                                    #calcula a GLCM
                                    gridZone = img_as_ubyte(gridZone)
                                    glcm = greycomatrix(gridZone, [1], [0], symmetric=False, normed=True)
                                    featureMatrix[iX, iY, iOp, iPolyOrder] = \
                                    greycoprops(glcm, 'contrast')[0, 0]
                                elif Op in ["dissimilarity"]:
                                    #calcula a GLCM
                                    gridZone = img_as_ubyte(gridZone)
                                    glcm = greycomatrix(gridZone, [1], [0], symmetric=False, normed=True)
                                    featureMatrix[iX, iY, iOp, iPolyOrder] = \
                                    greycoprops(glcm, 'dissimilarity')[0, 0]
                                elif Op in ["homogeneity"]:
                                    #calcula a GLCM
                                    gridZone = img_as_ubyte(gridZone)
                                    glcm = greycomatrix(gridZone, [1], [0], symmetric=False, normed=True)
                                    featureMatrix[iX, iY, iOp, iPolyOrder] = \
                                    greycoprops(glcm, 'homogeneity')[0, 0]

                    elif nDimension == 3:
                        for iZ in range(nGridZ):
                            gridZone = datasetDic[iDataset][iX*xlength : (iX+1)*xlength, \
                                               iY*ylength : (iY+1)*ylength, \
                                               iZ*zlength : (iZ+1)*zlength]                
                            for iPolyOrder in range(npoly):
                                for iOp, Op in enumerate(typeOp):
                                    if Op in ["average", "Average", "mean"]:
                                        featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder] = \
                                        np.mean(gridZone)**(iPolyOrder+1)
                                        if np.isnan(featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder]) or \
                                           np.isinf(featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder]): 
                                               featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder] = 0.0
                                    elif Op in ["max", "Max"]:
                                        featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder] = \
                                        np.amax(gridZone)**(iPolyOrder+1)
                                    elif Op in ["min", "Min"]:
                                        featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder] = \
                                        np.amin(gridZone)**(iPolyOrder+1)
                                    elif Op in ["variance", "var", \
                                                "Expectation", "expectation"]:
                                        featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder] = \
                                        np.var(gridZone)**(iPolyOrder+1)
                                    elif Op in ["covariance", "cov"]:
                                        featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder] = \
                                        np.cov(gridZone)**(iPolyOrder+1)
                                    elif Op in ["sum", "Sum"]:
                                        featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder] = \
                                        np.sum(gridZone)**(iPolyOrder+1)
                                       
            # We flatten the 4D/5D featureMatrix:
            features[iDataset,:] = featureMatrix.flatten()
            
        return features
                            
###############################################################################          
###############################################################################      
            
class Prediction:
    
    def __init__(self, featuresMatrix, label=[]):
        self.features = featuresMatrix
        self.label = label
        # Number of elements in the dataset used for computing the features 
        # matrix and number of features computed:
        (self.nSamples, self.nFeatures) = featuresMatrix.shape
        
        
    def datasetSplit(self, ratioSplit = 0.8):		
        # Number of samples in the entire dataset:		
        nbSamples = self.nSamples		
    		
        # Number of images used for training:		
        nTrain = round(ratioSplit * nbSamples)		
    		
        # Number of images used for validation:		
        # nValid = nbSamples - nTrain		
    		
        shuffleIndex = np.arange(nbSamples)		
        np.random.shuffle(shuffleIndex)		
    		
        # Indices of the training and validation datasets:		
        trainIndex = shuffleIndex[:nTrain-1]		
        validIndex = shuffleIndex[nTrain:]		
        indexSplit = {"training":trainIndex, "validation":validIndex}		
    		
#       print ("Training and validation dataset indexes created. Average age:\n\		
#       Training dataset: {} y.o\nvalidation dataset: {} y.o".format(int(round(labelTrain.mean())), \		
#       int(round(labelValid.mean()))))	

        labelTraining=[]
        labelValidation=[]
        if len(self.label) != 0:	
            labelTraining = self.label[indexSplit["training"]]
            labelValidation = self.label[indexSplit["validation"]]		
    		
        return indexSplit, labelTraining, labelValidation		
    		
        		
    def featureSplit (self, ratioSplit=0.8, **indexSplit):		
        if len(indexSplit) == 0:		
            indexSplit = self.datasetSplit(int(ratioSplit)	)	
        		
        # Labels of the training and validation datasets:		
        featureDic = {}		
        featureDic["training"] = self.features[indexSplit["training"], :]		
        featureDic["validation"] = self.features[indexSplit["validation"], :]		
        return featureDic
        
    
    def modelParameters(self, featureDic=[], shrinkageParameter = 0, technique = "LS"):
        # Number of elements in the dataset used for computing the features 
        # matrix and number of features computed:
        nbSamples = self.nSamples
        nbFeatures = self.nFeatures

        if len(featureDic) == 0: 		
            featureDic = {"training": self.features}		
            
        # Choice of the method:
        if shrinkageParameter == 0:
            technique = "LS"
        else:
            technique = "Ridge"
            
        if technique in ["Least square", "least square", "LS"]:
            featureMatrix = featureDic["training"]
        elif technique in ["Ridge", "ridge", "Ridge regression", "ridge regression"]:
            featureMatrix = np.zeros([ nbSamples + nbFeatures, nbFeatures ])
            featureMatrix[:nbSamples-1,:] = featureDic["training"]
            # The lower part of the feature matrix becomes the identity matrix
            # times the shrinkage parameter:
            featureMatrix[nbSamples,:] = np.sqrt(shrinkageParameter)* \
                np.identity(nbFeatures)     

        label = np.concatenate((self.label,np.zeros(nbFeatures)))
        
        # To compute the parameters of the model, minimizing the least square 
        # of sum(|y-ax|²), the function lstsq is used: 
        parameters,sumResiduals,rank,singularValues = np.linalg.lstsq(featureMatrix, label)
        
        return parameters
        
        
    def buildClassifier( self, featureDic=[], labelTraining=[], classifier = "LASSO"):
        
        label = labelTraining	
        featureMatrix = featureDic["training"]
        #parameters of classifiers:
        #copy_X: to copy the input data and do not overwrite
        #n_jobs: number of jobs used for computation. -1 means all CPUs will be used
        #cv: method for cross-validation. None means use Leave-one-out
        
        if classifier == "Linear regression":
            clf = linear_model.LinearRegression(copy_X=True, n_jobs=-1, \
                                                normalize=False)
        elif classifier == "LASSO":
            clf = linear_model.Lasso(alpha=2000, copy_X=True, \
                                     normalize=False, tol=0.1)
        elif classifier == "Ridge":
            clf = linear_model.Ridge(alpha=2000, copy_X=True, \
                                     normalize=False, solver='auto', tol=0.1)
        elif classifier == "RidgeCV": #ridge cross validation
            clf = linear_model.RidgeCV(alphas=[0.1, 0.5, 1.0, 10.0], cv=10, \
                                       fit_intercept=True, normalize=True )
        elif classifier == "SVR-RBF": #support vector regression with Radial Basis Function kernel
            clf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        elif classifier == "SVR-Linear": #support vector regression with linear kernel
            clf = LinearSVR( C=1, epsilon=0 )
        elif classifier == "GaussianNB":
            clf = GaussianNB()
            
        clf.fit( featureMatrix, label )
        
        parameters =[]
        #if classifier != "SVR-RBF" and classifier != "GaussianNB":
        #    parameters = clf.coef_
        #    parameters = parameters.reshape( clf.coef_.size, 1)
        #    print("calculou coef")
                
        return clf, parameters
        
            
    def predict(self, parameters, featureDic=[], labelValidation=[], clf=[]):
        # Number of elements in the dataset used for computing the features 
        # matrix and number of features computed:
        features = featureDic["validation"]
        nbSamples = features.shape[0]
        
        # Prediction of the model for the given dataset:
        if len(parameters) == 0:
            predictedData = clf.predict(features)
            print("clf predict")
        else:
            predictedData = features.dot( parameters )
            
        #predictedData = np.rint(predictedData)
        
        print("Predicted data: ", predictedData.shape)
        
        if len(labelValidation) != 0:
            
            # Computation of the mean squared error of the predicted data:
            MSE = round(np.mean((predictedData - labelValidation)**2)) 
            
            print("The achieved score is: " + str(MSE))
            
            print( " MSE: ", mean_squared_error( labelValidation, predictedData ) )
            
            # we sort the label (ascending order) and the predicted data:
            indexSort = np.argsort(labelValidation)
            labelSort = np.array(labelValidation[indexSort])
            predictedDataSort = np.array(predictedData[indexSort])
            
            # X-axis:
            x = np.linspace(1, nbSamples, nbSamples)
            print("number of samples: ", nbSamples)
            
            import matplotlib.pyplot as plt
            import pylab
            plt.figure(100)
            feature0 = features[:,0]
            plt.plot(x, feature0, color="green", linewidth=1, \
                     linestyle='--', marker='o')
            plt.plot(x, predictedDataSort, color="blue", linewidth=1, \
                     linestyle='--', marker='o')
            plt.plot(x, labelSort, color="red", linewidth=1, \
                     linestyle='--', marker='o')
            plt.title("Validation of the model")
            plt.xlabel("Patient number")
            plt.ylabel("Age")
            
            pylab.show()
            
        return predictedData
        
                    
            

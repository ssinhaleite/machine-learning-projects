# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:06:58 2016

@author: valentin
"""

import numpy as np

from sklearn import linear_model

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
                    
        # In the following loop, we call the functions allowing to compute the \
        # features chosen by the users in the dictionary featureType:
        for k,v in featureType.items():
            if isinstance(v,dict): 
                localFeatures = functionDic[k](self.dataset, **v)
            else:
                if v < 1:
                    v = 1
                localFeatures = functionDic[k](self.dataset, npoly=v)
                
            featureMatrix = np.concatenate((featureMatrix, localFeatures), axis=1)
            
            return featureMatrix
            
###############################################################################          
            
    def gridOperation(self, nGrid=(10,10), npoly=1, type2D="center", axis=2, \
                      *typeOp):
   
        # Dictionary containing the dataset:
        datasetDic = self.dataset
        
        if npoly < 1:
            npoly = 1
            
        if len(nGrid) < 2:
            nDimension = 2
        else:
            nDimension = 3
            
        if isinstance(typeOp,list) == False:
            typeOp = [typeOp, ]
        nOp = len(typeOp)
        
        # Length (in points) of the grid along the different dimension:     
        xlength = round(self.sizeX/nGrid[0])
        ylength = round(self.sizeY/nGrid[1])
        
        if nDimension == 3:
            zlength = round(self.sizeZ/nGrid[2])
        
        # Number of subdivisions of the grid along each dimension: 
        nGridX = np.ceil(self.sizeX / xlength) 
        nGridY = np.ceil(self.sizeY / ylength) 
        if nDimension == 3:
            nGridZ = np.ceil(self.sizeZ / zlength) 
        
        # Creation of the featureMatrix containing the features in a 4D/5D matrix:
        if nDimension == 2:
            featureMatrix = np.empt([nGridX, nGridY, nOp, npoly])
            features = np.empt([nGridX*nGridY*nOp*npoly, self.nData])
        elif nDimension == 3:
            featureMatrix = np.empt([nGridX, nGridY, nGridZ, nOp, npoly])
            features = np.empt([nGridX*nGridY*nGridZ*nOp*npoly, self.nData])

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
                    elif axis == 1:
                        image2D = datasetDic[iDataset][:,:,self.Z0]
                elif type2D == "sum":
                    image2D = datasetDic[iDataset].sum(axis)
                else:
                    if axis == 0:
                        image2D = datasetDic[iDataset][type2D,:,:]
                    elif axis == 1:
                        image2D = datasetDic[iDataset][:,type2D,:]
                    elif axis == 1:
                        image2D = datasetDic[iDataset][:,:,type2D]
                
            for iX in range(nGridX):
                for iY in range(nGridY):
                    if nDimension == 2:
                        gridZone = image2D[iX*xlength : (iX+1)*xlength, \
                                           iY*ylength : (iY+1)*ylength]                
                        for iPolyOrder in range(npoly):
                            for Op in typeOp:
                                 if Op in ["average", "Average", "mean"]:
                                     featureMatrix[iX, iY, Op, iPolyOrder] = \
                                     np.mean(gridZone)**(iPolyOrder+1)
                                 elif Op in ["max", "Max"]:
                                     featureMatrix[iX, iY, Op, iPolyOrder] = \
                                     np.amax(gridZone)**(iPolyOrder+1)
                                 elif Op in ["min", "Min"]:
                                     featureMatrix[iX, iY, Op, iPolyOrder] = \
                                     np.amin(gridZone)**(iPolyOrder+1)
                                 elif Op in ["variance", "Var", "Expectation", \
                                             "expectation"]:
                                     featureMatrix[iX, iY, Op, iPolyOrder] = \
                                     np.var(gridZone)**(iPolyOrder+1)
                    elif nDimension == 3:
                        for iZ in range(nGridZ):
                            gridZone = image2D[iX*xlength : (iX+1)*xlength, \
                                               iY*ylength : (iY+1)*ylength, \
                                               iZ*zlength : (iZ+1)*zlength]                
                            for iPolyOrder in range(npoly):
                                for Op in typeOp:
                                    if Op in ["average", "Average", "mean"]:
                                        featureMatrix[iX, iY, iZ, Op, \
                                                      iPolyOrder] = \
                                        np.mean(gridZone)**(iPolyOrder+1)
                                    elif Op in ["max", "Max"]:
                                        featureMatrix[iX, iY, iZ, Op, \
                                                      iPolyOrder] = \
                                        np.amax(gridZone)**(iPolyOrder+1)
                                    elif Op in ["min", "Min"]:
                                        featureMatrix[iX, iY, iZ, Op, \
                                                      iPolyOrder] = \
                                        np.amin(gridZone)**(iPolyOrder+1)
                                    elif Op in ["variance", "Var", \
                                                "Expectation", "expectation"]:
                                        featureMatrix[iX, iY, iZ, Op, \
                                                      iPolyOrder] = \
                                        np.var(gridZone)**(iPolyOrder+1)
                                       
            # We flatten the 4D/5D featureMatrix:
            features[iDataset,:] = featureMatrix.flatten()
            
        return features
                            
###############################################################################          
###############################################################################      
            
class Prediction:
    
    def __init__(self, featuresMatrix):
        self.features = featuresMatrix
        
        # Number of elements in the dataset used for computing the features 
        # matrix and number of features computed:
        (self.nSamples, self.nFeatures) = featuresMatrix.shape
        
    
    def modelParameters(self, label, shrinkageParameter = 0, technique = "LS"):
        # Number of elements in the dataset used for computing the features 
        # matrix and number of features computed:
        nbSamples = self.nSamples
        nbFeatures = self.nFeatures
            
        # Choice of the method:
        if shrinkageParameter == 0:
            technique = "LS"
            
        if technique in ["Least square", "least square", "LS"]:
            featureMatrix = self.features
        elif technique in ["Ridge", "ridge", "Ridge regression", "ridge regression"]:
            featureMatrix = np.zeros([ nbSamples + nbFeatures, nbFeatures ])
            featureMatrix[:nbSamples-1,:] = self.features
            # The lower part of the feature matrix becomes the identity matrix
            # times the shrinkage parameter:
            featureMatrix[nbSamples,:] = np.sqrt(shrinkageParameter)* \
                np.identity(nbFeatures)     
            label = np.concatenate((label,np.zeros(nbFeatures)))
        
        # To compute the parameters of the model, minimizing the least square 
        # of sum(|y-ax|²), the function lstsq is used: 
        parameters,sumResiduals,rank,singularValues = np.linalg.lstsq(featureMatrix, label)
        
        return parameters
        
        
    def buildClassifier( self, labels, classifier = "LASSO"):
        
        featureMatrix = self.features
        
        #parameters of classifiers:
        #copy_X: to copy the input data and do not overwrite
        #n_jobs: number of jobs used for computation. -1 means all CPUs will be used
        #cv: method for cross-validation. None means use Leave-one-out
        
        if classifier == "Linear regression":
            clf = linear_model.LinearRegression(copy_X=True, n_jobs=-1, \
                                                normalize=True)
        elif classifier == "LASSO":
            clf = linear_model.Lasso(alpha=0.1, copy_X=True, \
                                     normalize=True, tol=0.0001)
        elif classifier == "Ridge":
            clf = linear_model.Ridge(alpha=0.5, copy_X=True, \
                                     normalize=True, solver='lsqr', tol=0.001)
        elif classifier == "RidgeCV": #ridge cross validation
            clf = linear_model.RidgeCV( alphas=[0.1, 1.0, 10.0], cv=None, \
                     normalize=True )

        clf.fit( featureMatrix, labels )       
        return clf, clf.coef_
        
            
    def predict(self, parameters, label="no label"):
        # Number of elements in the dataset used for computing the features 
        # matrix and number of features computed:
        nbSamples = self.nSamples
        
        # Prediction of the model for the given dataset:
        predictedData = parameters * self.features
        
        if isinstance(label, np.array):
            
            # Computation of the mean squared error of the predicted data:
            MSE = round(np.mean((predictedData - label)**2), 2) 
            
            print("The achieved score is: " + str(MSE))
            
            # we sort the label (ascending order) and the predicted data:
            indexSort = np.argsort(label)
            labelSort = label(indexSort)
            predictedDataSort = predictedData(indexSort)
            
            # X-axis:
            x = np.linespace(1, nbSamples, nbSamples)
            
            import matplotlib.pyplot as plt
            plt.figure(100)
            plt.plot(x, predictedDataSort, color="blue", linewidth=1, \
                     linestyle='--', marker='o')
            plt.plot(x, labelSort, color="blue", linewidth=1, \
                     linestyle='--', marker='o')
            plt.title("Validation of the model")
            plt.xlabel("Patient number")
            plt.ylabel("Age")
            
        return predictedData
        
                    
            
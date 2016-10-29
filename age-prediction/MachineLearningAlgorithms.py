# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:06:58 2016

@author: valentin
"""
# Import the needed libraries:
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

from sklearn import linear_model
from skimage.feature import greycomatrix, greycoprops
from skimage import img_as_ubyte

class Features:
    """ The objects of this class are used to extract the features from a
    3D matrix in order to apply some machine learning alorithm
    
    CLASS ATTRIBUTES:
        dataset -- dictionary, contain the dataset whose the keys are the 
        indices of the 3D MRI images of the dataset and the values are a 3D  
        np.array containing these 3D images
        ndata -- integer, number of 3D images in the dataset
        sizeX -- integer, number of voxels along the 1st dimension of the 3D 
        np.array
        sizeY -- integer, number of voxels along the 2nd dimension of the 3D 
        np.array
        sizeZ -- integer, number of voxels along the 3rd dimension of the 3D 
        np.array
        nVoxels -- integer, number of voxels contained in the 3D images
        X0 -- integer, central voxel along the 1st dimension of the 3D 
        np.array
        Y0 -- integer, central voxel along the 1st dimension of the 3D 
        np.array
        Z0 -- integer, central voxel along the 1st dimension of the 3D 
        np.array
    """
    
    def __init__(self, datasetDic):
        """
        Constructuctor of the class Features
        """
        
        # Dictionary 
        self.dataset = datasetDic
        self.nData =  len( self.dataset ) 
        
        # number of voxels along the 1st dimension of the 3D np.array
        (self.sizeX, self.sizeY, self.sizeZ) = self.dataset[0].shape
        
        # Number of voxels:
        self.nVoxels = self.sizeX * self.sizeY * self.sizeZ
        
        # Central positions along x, y, z:
        self.X0 = int(float(self.sizeX)/2)
        self.Y0 = int(float(self.sizeY)/2)
        self.Z0 = int(float(self.sizeZ)/2)
        
        
    def featureExtraction(self, **featureType):
        """
        DEFINITION:
        This function computes the feature matrix for a dataseet. This can be 
        used either to determine the parameter matrix of our model if we
        have labeled data (= training dataset) or to realize prediction (= 
        validation dataset or test dataset).
        
        INPUT VARIABLES:
        featureType: dictionary. It should respect the following construction 
        rules:
            1) The key is the name of the function (string) that we want to use 
            to extract the features we wish. 
            2) The value represents the parameters of the used function knowing
            that:
                - the parameters can be either a dictionary: {"input1": value1,
                ..., "inputN": valueN} or a number. If it is a number it will 
                be interpreted as the polynomial order on which we want to fit 
                the given feature
        """
        # Initialisation of the featureMatrix which will contain the values  \
        # of all the features computed for each element of the dataset:
        # featureMatrix = np.ones([self.nData,1]) # This corresponds to the bias
        it = 0
        
        # Database dictionary containing the functions potentially used for 
        # features extraction:
        functionDic = {} # should be fulfill as soon as we add a new function
                         # for feature extraction !!!
        functionDic["gridOperation"] = self.gridOperation
        functionDic["threshold"] = self.threshold            
        
        # In the following loop, we call the functions allowing to compute the \
        # features chosen by the users in the dictionary featureType:
        for k,v in featureType.items():
            if isinstance(v,dict): 
                localFeatures = functionDic[k](**v)
            else:
                if v < 1:
                    v = 1
                localFeatures = functionDic[k](npoly=v)	
             
            if it == 0:
                featureMatrix = localFeatures
                it += 1
            else:
                featureMatrix = np.concatenate((featureMatrix, localFeatures), axis=1)
                
            
            return featureMatrix
            
###############################################################################          
            
    def gridOperation(self, typeOp=["mean"], nGrid=(10,10,"center"), npoly=1):
        """
        To divide the 3D MRI image into smaller volumes and then applied some
        mathematical operation on these volumes (mean, variance, max...)
        
        INPUTS:
            typeOp -- list of strings, type of operations which will be applied 
            on the smaller volume
            nGrid -- tuple of 2 or 3 integers, number of space division we want
            along the two or three dimensions.
            npoly -- integer, polynomial order of the features. 
            
        OUTPUTS:
            features -- 2D np.array of float numbers, the feature matrix with
            the features computed for each 3D images
            
        
        Example of use:
        2D) 
        featureDic = {"gridOperation": { "nGrid":(15,15,"center"), "npoly":2, \
            "typeOp":["mean", "var"]} } 
        
        3D) 
        featureDic = {"gridOperation": { "nGrid":(15,15,15), "npoly":2, \
            "typeOp":["mean", "var"]} } 
        """
        
        # Dictionary containing the dataset:
        datasetDic = self.dataset
        
        # Number of operations required by the user:
        nOp = len(typeOp)
        
        # Polynomial order of the computed features:
        if npoly < 1:
            npoly = 1
       
        # In this loop, we determine the number of dimension along sith the 
        # space will be divided
        if len(nGrid) == 2:
            nDimension = 2
            type2D = ["center"]
        else:
            nDimension = 3
            for i in range(2):
                if isinstance(nGrid[i], str):
                    nDimension = 2
                    axis = i
                    type2D = [nGrid[i]]                                  
            
        # Number of subdivisions of the grid along each dimension: 
        nGridX = nGrid[0]
        nGridY = nGrid[1]
        if nDimension == 3:
            nGridZ = nGrid[2]
            
        # Length (in points) of the grid along the different dimension:
        xlength = []
        ylength  =[]
        xlength.append(0)
        ylength.append(0)
        
        xGrid = 0
        yGrid = 0
        
        for i in range(nGridX):
            xGrid += int(round((self.sizeX-xGrid)/(nGridX-i)))
            xlength.append(xGrid)
            
        for i in range(nGridY):
            yGrid += int(round((self.sizeY-yGrid)/(nGridY-i)))
            ylength.append(yGrid)
        
        if nDimension == 3:
            zlength =[]
            zlength.append(0)
            zGrid = 0
            for i in range(nGridZ):
                zGrid += int(round((self.sizeZ-zGrid)/(nGridZ-i)))
                zlength.append(zGrid)

        # Creation of the featureMatrix containing the features in a 4D/5D matrix:
        if nDimension == 2:
            featureMatrix = np.empty([nGridX, nGridY, nOp, npoly])
            features = np.empty([self.nData, nGridX*nGridY*nOp*npoly])
        elif nDimension == 3:
            featureMatrix = np.empty([nGridX, nGridY, nGridZ, nOp, npoly])
            features = np.empty([self.nData, nGridX*nGridY*nGridZ*nOp*npoly])
        
        # Status bar:
        # Counts of the number of steps already processed:
        CalculDone = 1
        statusBar =""
        sampleStatus = 1 # in %
        nbSampleStatus = int(100 / sampleStatus)
        
        for iDataset in range(self.nData):
            if iDataset == 0:
                startTime = time.time()
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
                        gridZone = image2D[xlength[iX] : xlength[iX+1], \
                                           ylength[iY] : ylength[iY+1]]                
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
                    elif nDimension == 3:
                        for iZ in range(nGridZ):
                            gridZone = datasetDic[iDataset][ \
                                            xlength[iX] : xlength[iX+1], \
                                            ylength[iY] : ylength[iY+1], \
                                            zlength[iZ] : zlength[iZ+1]] 
               
                            for iPolyOrder in range(npoly):
                                for iOp, Op in enumerate(typeOp):
                                    if Op in ["average", "Average", "mean"]:
                                        featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder] = \
                                        np.mean(gridZone)**(iPolyOrder+1)
                                        
                                    elif Op in ["max", "Max"]:
                                        featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder] = \
                                        np.amax(gridZone)**(iPolyOrder+1)
                                        
                                    elif Op in ["min", "Min"]:
                                        featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder] = \
                                        np.amin(gridZone)**(iPolyOrder+1)
                                        
                                    elif Op in ["contrast3D", "Michelson"]:
                                        minGrid = np.amin(gridZone)
                                        maxGrid = np.amax(gridZone)
                                        if minGrid == 0 and minGrid == 0:
                                            michelsonContrast = 0
                                        else:
                                            michelsonContrast = \
                                            (maxGrid-minGrid)/(maxGrid+minGrid)
                                        
                                        featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder] = \
                                        michelsonContrast**(iPolyOrder+1)
                                        
                                    elif Op in ["variance", "var", \
                                                "Expectation", "expectation"]:
                                        featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder] = \
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
             
            # Status bar:       
            status = (100 * (iDataset+1) / (self.nData))
            if iDataset == 0:
                endTime = time.time()
                time1Iteration = endTime - startTime
                totalTime = 100 * time1Iteration / status
                
            if status > CalculDone * sampleStatus:
                    CalculDone += 1
                    statusBar += "="
                    remainingTime = int(round(totalTime * (1 - CalculDone*sampleStatus/100))) 
                    #print("\r" + statusBar)
                    sys.stdout.write("\r" + "=" * CalculDone + \
                         " " * (nbSampleStatus-CalculDone) + str(remainingTime)+"s" )
                    sys.stdout.flush() 
             
            
            # We flatten the 4D/5D featureMatrix:
            features[iDataset,:] = featureMatrix.flatten()
        
         
        return features

###############################################################################        

    def threshold(self, nLevel=10, thresholdType = "Energy", axis=-1):
        """
        THIS DOES NOT WORK PROPERLY. IT SHOULD NOT BE USED !!
        """
        # Dictionary containing the dataset:
        datasetDic = self.dataset
        
        # Status bar:
        # Counts of the number of steps already processed:
        CalculDone = 1
        statusBar =""
        sampleStatus = 1 # in %
        nbSampleStatus = int(100 / sampleStatus)
        
        for iDataset in range(self.nData):
            if iDataset == 0:
                startTime = time.time()
            if thresholdType ==  "Energy":
                # 3D image:
                image3D = datasetDic[iDataset]
                
                # Computation of the total energy:
                energy = np.sum(image3D)
                
                # Mean energy:
                energyMean = energy / self.nVoxels
                
                # Factor for the upper threshold level:
                factorEnergy = 5
                maxLevel = factorEnergy * energyMean
                
                # Threshold vector:
                levelVector = (np.arange(nLevel)+1) * maxLevel / nLevel
                
                belowThreshold = 0
                           
                if axis in (0,-1): #We work on al the image YZ image
                    # For each level of threshold we compute the number of voxels
                    # between level(i) and level(i+1):
                    thresholdVector = np.empty([self.sizeX, nLevel])    
                    
                    # Creation of the feature matrix:
                    features = np.empty([self.nData, (nLevel-1)*self.sizeX])
                    
                    # For each image we compute the threshold:
                    for iImage in range(self.sizeX):
                        # The image 2D:
                        image2D = image3D[iImage,:,:]
                                                   
                        for i, level in enumerate(levelVector):
                            thresholdVector[iImage, i] = - belowThreshold + \
                                len(np.where(image2D<level)[0])
                            belowThreshold += thresholdVector[iImage, i]
                        features[iDataset,:] = thresholdVector[:,1:].flatten()
                                                  
#                        for k in range (self.sizeY):
#                            for n in range (self.sizeZ):
#                                for i, level in enumerate(levelVector):
#                                    if image2D[k,n] < level :
#                                        thresholdVector[iImage, i] += 1
#                                        break
#                        for j in range(nLevel-1, 0, -1):
#                            thresholdVector[iImage, j] -= thresholdVector[iImage, j-1]

                    if axis == -1:
                        features[iDataset,:] = 100*thresholdVector[:, 1:].flatten() /self.nVoxels
                    else:
                        features[iDataset,:] = 100*np.sum(thresholdVector[:, 1:], axis=0) /self.nVoxels
                                
                    status = (100 * (iDataset+1) / (self.nData))
                    if iDataset == 0:
                        endTime = time.time()
                        time1Iteration = endTime - startTime
                        totalTime = 100 * time1Iteration / status
                        
                    if status > CalculDone * sampleStatus:
                            CalculDone += 1
                            statusBar += "="
                            remainingTime = int(round(totalTime * (1 - CalculDone*sampleStatus/100))) 
                            #print("\r" + statusBar)
                            sys.stdout.write("\r" + "=" * CalculDone + \
                                 " " * (nbSampleStatus-CalculDone) + str(remainingTime)+"s" )
                            sys.stdout.flush() 
                    
                elif axis == 1: #We work on al the image XZ image
                    # For each level of threshold we compute the number of voxels
                    # between level(i) and level(i+1):
                    thresholdVector = np.empty([self.sizeY, nLevel])    
                    
                    # Creation of the feature matrix:
                    features = np.empty([self.nData, nLevel])
                    
                    for iImage in range(self.sizeY):
                        # The image 2D:
                        image2D = image3D[:,iImage,:]
                                                   
                        for i, level in enumerate(levelVector):
                            thresholdVector[iImage, i] = - belowThreshold + \
                                np.len(np.where(image2D<level)[0])
                            belowThreshold += thresholdVector[iImage, i]
                        features[iDataset,:] = thresholdVector.flatten()
                
                elif axis == 2: #We work on al the image XY image
                    # For each level of threshold we compute the number of voxels
                    # between level(i) and level(i+1):
                    thresholdVector = np.empty([self.sizeZ, nLevel])    
                    
                    # Creation of the feature matrix:
                    features = np.empty([self.nData, nLevel])
                    
                    for iImage in range(self.sizeZ):
                        # The image 2D:
                        image2D = image3D[:,:,iImage]
                                                   
                        for i, level in enumerate(levelVector):
                            thresholdVector[iImage, i] = - belowThreshold + \
                                np.len(np.where(image2D<level)[0])
                            belowThreshold += thresholdVector[iImage, i]
                        features[iDataset,:] = thresholdVector.flatten()

        return features


                      
###############################################################################          
###############################################################################      
            
class Prediction:
    """ The objects of this class are used to determine the parameters of our
    model from the training dataset using a cross-validation stage. It can also
    be used to predict the labels of the test set once the model parameters
    determined
    
    CLASS ATTRIBUTES:
        features -- 2D np.array of float numbers, the feature matrix with
        the features computed for each 3D images
        label -- list of integers, the labels of the training dataset
        nSamples -- integer, number of 3D MRI images in the dataset
        nFeatures -- integer, number of computed features
        
    """
    def __init__(self, featuresMatrix, label=[]):
        """
        Constructuctor of the class Prediction
        """
        
        # 2D feature matrix extracted from the original 3D images:
        self.features = featuresMatrix
        
        # Labels of the training dataset:
        self.label = label
        
        # Number of elements in the dataset used for computing the features 
        # matrix and number of features computed:
        (self.nSamples, self.nFeatures) = featuresMatrix.shape
                
    
    def modelParameters(self, featureDic=[], shrinkageParameter = 0, technique = "LS"):
        """
        DEPRECATED: you should use buildclassifier instead. The simple ridge 
        regression gives exactly the same results as the one in the 
        buildclassifier function but slower. This is why we recommand to use
        buildclassifier
        """
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
        
        
    def buildClassifier( self, featureTraining, label, method = "LASSO"):
        """ To compute the model parameters given the feature matrix and the 
        labels of a training dataset
        
        INPUTS: 
            featureTraining -- 2D np.array, feature matrix containung the 
            features extracting from the dataset
            method -- string, linear method used for determining the parameters
            of the model
            
        OUTPUTS:
            parameters: 1D np.array, the parameters of the model
        """
        
        # Labels of the dataset:
        # label = self.label	
        
        #parameters of classifiers:
        #alpha: regularizer (or shrinkage) parameter
        #copy_X: to copy the input data and do not overwrite
        #n_jobs: number of jobs used for computation. -1 means all CPUs will be used
        #cv: method for cross-validation. None means use Leave-one-out
        
        if method == "Linear regression":
            clf = linear_model.LinearRegression(copy_X=True, n_jobs=-1, \
                                                normalize=True)
        elif method == "LASSO":
            clf = linear_model.Lasso(alpha=0.1, copy_X=True, \
                                     normalize=True, tol=0.0001)
        elif method == "Ridge":
            clf = linear_model.Ridge(alpha=0.5, copy_X=True, \
                                     normalize=True, solver='lsqr', tol=0.001)
        elif method == "RidgeCV": #ridge cross validation
            clf = linear_model.RidgeCV( alphas=[0.0, 0.1, 10.0], cv=None, \
                                       fit_intercept=True, normalize=True )
        
        label = label.reshape( label.size, 1 )
        
        # Use the function fit from the sklearn module
        clf.fit( featureTraining, label ) 
        
        # Use the function coef_ from the sklearn module to compute the 
        # parameters of our model:
        parameters = clf.coef_
        parameters = parameters.reshape(clf.coef_.size, 1)
        parameters = np.append( clf.intercept_, parameters) # to add the bias
        
        return parameters
        
            
    def predict(self, parameters, features, labelValidation=[]):
        """ To compute the model parameters given the feature matrix and the 
        labels of a training dataset
        
        INPUTS:
            parameters: 1D np.array, the parameters of the model
            features -- 2D np.array, feature matrix containung the 
            features extracting from the dataset
            labelValidation -- 1D np.array, the labels used for validation
            
        OUTPUTS:
            predictedData: 1D np.array, the predicted labels
        """
        # Number of 3D MRI images in the dataset:
        nbSamples = features.shape[0]

        # The first column of the feature matrix should be the bias:
        bias = np.ones([nbSamples,1])
        features = np.concatenate((bias, features), axis=1)
        
        # Prediction of the model for the given dataset:
        predictedData = features.dot( parameters )
        
        # Computation of the mean squared error of the predicted data:
        if len(labelValidation) > 0:
            MSE = (np.mean((predictedData - labelValidation)**2)) 
            return predictedData, MSE
            
        return predictedData
        
                    
    def crossValidation(self, nFold=10, typeCV="random"):
        """ To compute the model parameters through cros-validation and 
        estimate the choice of the features by computing the mean-squared error
        
        INPUTS:
            nFold: integer, number of folds (or buckets) used for 
            cross-validation
            typeCV -- string, type of cross-validation
            
        OUTPUTS:
            MSE: float, mean-squared error computed after the cross-validation
        """
        
        featuresMatrix = self.features 
        label = self.label
        
        # Number of samples in the training dataset
        nSamples = self.nSamples
        
        # We shuffle the indices of the feature matrix in the first dimension 
        # (number of samples)
        if typeCV == "random":
            # Array with the indices of the sample in the dataset
            indices = np.arange(nSamples)
            
            # We shuffle the indices
            indicesShuffled = np.random.shuffle(indices)
            featuresMatrix = featuresMatrix[indicesShuffled, :][0,:,:]
        
        # Array with the indices of the sample in the dataset:
        indices = np.arange(nSamples)
        
        # Number of samples in one training fold:
        samplePerFoldTrain = int(np.floor(nSamples / nFold))
            
        # Number of samples used for training in cross validation:
        sampleTrain = (nFold-1) * samplePerFoldTrain

        # Number of samples used for training over all the cross validation 
        # process
        sampleTrainCV = nFold * samplePerFoldTrain

        # Number of samples used for validation
        sampleValid = nSamples - sampleTrain
        
        # Creation of the dictionary containing the indices of the samples used
        # for training for each bucket
        indexTrain = {}
            
        # Creation of the dictionary containing the indices of the samples used
        # for validation for each bucket
        indexValid = {}
        
        # Creation of the array containing the MSE error for each cross 
        # validation
        MSEArray = np.zeros([nFold])         
        
        # Matrix containing all the predictions:
        predictions = np.empty([sampleTrainCV])
        
        for i in range(nFold):
            
            # We divide the all training dataset into K buckets (or folds):
            indexValid = np.arange(i*samplePerFoldTrain, i*samplePerFoldTrain + sampleValid )
            indexTrain = np.delete(indices, indexValid)
            
            # We compute the model parameters
            parameters = self.buildClassifier(featuresMatrix[indexTrain, :], \
                            label[indexTrain], method = "RidgeCV")
            
            # We predict the data with the computed parameters
            predictedData, MSEArray[i] = self.predict(parameters, \
                          featuresMatrix[indexValid, :], labelValidation=label[indexValid])
  
            # We save the predicted data in the appropriate matrix:
            predictions[i*samplePerFoldTrain : (i+1)*samplePerFoldTrain] = predictedData[:samplePerFoldTrain]
            
        # We compute the overall mean-squared error:
        MSE = round(np.mean(MSEArray),1)
        
        # we sort the label (ascending order) and the predicted data:
        labelComparison = label[:sampleTrainCV]
        indexSort = np.argsort(labelComparison)
        labelSort = np.array(labelComparison[indexSort])
        predictedDataSort = np.array(predictions[indexSort])
        
        # Plot the predicted data and the true data:
        plt.figure(100)
        
         # X-axis:
        x = np.linspace(1, sampleTrainCV, sampleTrainCV)
            
        # Plot of the predicted labels:
        plt.plot(x, predictedDataSort, color="blue", linewidth=1, \
                 linestyle='--', marker='o')
        
        # Plot of the true labels:
        plt.plot(x, labelSort, color="red", linewidth=1, \
                 linestyle='--', marker='o')
       
        plt.title("Validation of the model")
        plt.xlabel("Patient number")
        plt.ylabel("Age")
            
        return MSE 
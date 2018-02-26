# Import the needed libraries:
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import imageAnalysis as ia

#for image manipulation
from skimage.feature import greycomatrix, greycoprops
from skimage import img_as_ubyte
from scipy import stats
#for regression
from sklearn import linear_model
#for classification
import sklearn.svm as svm
from sklearn.svm import SVC
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
# for multilabel classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier

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
                localFeatures, binEdge = functionDic[k](**v)
            else:
                if v < 1:
                    v = 1
                localFeatures, _ = functionDic[k](npoly=v)    
             
            if it == 0:
                featureMatrix = localFeatures
                it += 1
            else:
                featureMatrix = np.concatenate((featureMatrix, localFeatures), axis=1)
                
            
            return featureMatrix, binEdge
            
###############################################################################          
            
    def gridOperation(self, typeOp=["mean"], nGrid=(10,10,"center"), npoly=1,\
                      binEdges = []):
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
            axis = 2
            type2D = "center"
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

        # Computation of the non-linear bins for the histogram:
        if ("histogram" in typeOp) or ("subhistogram" in typeOp) : 
            if isinstance(binEdges, int):
                nBins = binEdges
                if "histogram" in typeOp:
                    binEdges = np.zeros(nBins+1)
                    for i in range(3):
                        image = datasetDic[1+i*10]
                        binList = self.binHistogram(image, nBins)
                        binEdges += binList
                    binEdges/=3
                else:
                    binEdges = np.zeros([nGridX, nGridY, nGridZ, nBins+1])
                    for i in range(3):
                        image = datasetDic[1+i*10]
                        binList = self.binHistogram(image, nBins, nGrid=nGrid,\
                                    axisLength=[xlength, ylength, zlength])
                        binEdges += binList
                    binEdges/=3
            else:
                if "histogram" in typeOp :
                    nBins = len(binEdges) - 1
                else:
                    nBins = binEdges.shape[3] - 1

        # Creation of the featureMatrix containing the features in a 4D/5D matrix:
        if nDimension == 2:
            featureMatrix = np.empty([nGridX, nGridY, nOp, npoly])
            features = np.empty([self.nData, nGridX*nGridY*nOp*npoly])
        elif nDimension == 3:
            if ("histogram" in typeOp) or ("subhistogram" in typeOp):
                histoMatrix = np.empty([nGridX, nGridY, nGridZ, nBins])
                histo = np.empty([self.nData, nGridX*nGridY*nGridZ*nBins])
            else:
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
                                elif Op in ["mode"]:
                                    featureMatrix[iX, iY, iOp, iPolyOrder] = \
                                    stats.mode((gridZone)**(iPolyOrder+1))
                                elif Op in ["energy"]:
                                    #calcula a GLCM
                                    gridZone = (gridZone-gridZone.min())/(gridZone.max() - gridZone.min())
                                    gridZone = img_as_ubyte(gridZone)
                                    glcm = greycomatrix(gridZone, [1], [0], symmetric=False, normed=False)
                                    featureMatrix[iX, iY, iOp, iPolyOrder] = \
                                    greycoprops(glcm, 'energy')[0, 0]
                                elif Op in ["contrast"]:
                                    #calcula a GLCM
                                    gridZone = img_as_ubyte(gridZone)
                                    glcm = greycomatrix(gridZone, [1], [0], symmetric=False, normed=False)
                                    featureMatrix[iX, iY, iOp, iPolyOrder] = \
                                    greycoprops(glcm, 'contrast')[0, 0]
                                elif Op in ["dissimilarity"]:
                                    #calcula a GLCM
                                    gridZone = img_as_ubyte(gridZone)
                                    glcm = greycomatrix(gridZone, [1], [0], symmetric=False, normed=False)
                                    featureMatrix[iX, iY, iOp, iPolyOrder] = \
                                    greycoprops(glcm, 'dissimilarity')[0, 0]
                                elif Op in ["homogeneity"]:
                                    #calcula a GLCM
                                    gridZone = img_as_ubyte(gridZone)
                                    glcm = greycomatrix(gridZone, [1], [0], symmetric=False, normed=False)
                                    featureMatrix[iX, iY, iOp, iPolyOrder] = \
                                    greycoprops(glcm, 'homogeneity')[0, 0]
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

                                    if Op in ["median"]:
                                        featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder] = \
                                        np.median(gridZone)**(iPolyOrder+1)
                                        
                                    elif Op in ["max", "Max"]:
                                        featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder] = \
                                        np.amax(gridZone)**(iPolyOrder+1)
                                        
                                    elif Op in ["min", "Min"]:
                                        featureMatrix[iX, iY, iZ, iOp, \
                                                      iPolyOrder] = \
                                        np.amin(gridZone)**(iPolyOrder+1)
                                        
                                    elif Op == "histogram":
                                        histoLocal,_ = np.histogram(gridZone.flatten(),\
                                                         binEdges)
                                        histoMatrix[iX, iY, iZ, :] = histoLocal
                                            
                                    elif Op == "subhistogram":
                                        histoLocal,_ = np.histogram(gridZone.flatten(),\
                                                         binEdges[iX, iY, iZ, :])
                                        histoMatrix[iX, iY, iZ, :] = histoLocal
                                        
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
             
            # Status bar:       
            status = (100 * (iDataset+1) / (self.nData))
            if iDataset == 0:
                endTime = time.time()
                time1Iteration = endTime - startTime
                totalTime = 100 * time1Iteration / status
                
            if status > CalculDone * sampleStatus:
                    CalculDone += 1
                    #statusBar += "="
                    remainingTime = int(round(totalTime * (1 - CalculDone*sampleStatus/100))) 
                    #print("\r" + statusBar)
                    sys.stdout.write("\r" + "=" * CalculDone + \
                         " " * (nbSampleStatus-CalculDone) + str(remainingTime)+"s" )
                    sys.stdout.flush() 
             
            
            # We flatten the 4D/5D featureMatrix:
            if ("histogram" in typeOp) or ("subhistogram" in typeOp):
                histo[iDataset,:] = histoMatrix.flatten()
                if len(typeOp) > 1:
                    features[iDataset,:] = featureMatrix.flatten()
            else:
                features[iDataset,:] = featureMatrix.flatten()
        
        if ("histogram" in typeOp) or ("subhistogram" in typeOp):
            if len(typeOp) == 1:
                return histo, binEdges
            else:
                return np.append(features, histo), binEdges
        else:
            return features, 1
  
            
    def binHistogram(self,image, nBins, nGrid=[], axisLength=[]):
        
        if len(axisLength) > 2:
            typeBins="subhistogram"
        else:
            typeBins="histogram"
        
        if typeBins == "histogram":
            # we transform the 3D matrix into a vector:
            arrayBig = image.flatten()
            
            # lenght of the non-zero array:
            arrayLenght = len(arrayBig)
            
            # We sort the array:
            arraySort = np.sort(arrayBig)
            
            # Computation of the bin edges:
            binEdges = [arraySort[0]]
    
            # We look for the first non-zero element of the arrayBig:
            minIndex = np.argmin(arraySort[::-1])    
            non0Index = arrayLenght - (minIndex)
            
            
            binEdges.append(arraySort[non0Index])
             
            # Array with non zeros elements:
            arrayNonZeros = arraySort[non0Index:]
    
            # lenght of the non-zero array:
            arrayLenght = len(arrayNonZeros)
            
            # Number of points per bins:
            nPtsBins = int(round(arrayLenght / (nBins-1)))      
                
            for i in range(1, nBins):
                if i != nBins-1:
                    binEdges.append(arrayNonZeros[i*nPtsBins])
                else:
                    binEdges.append(arrayNonZeros[-1])
                
            binEdges[nBins] *= 1.5

        else:
            # Number of subdivisions of the grid along each dimension: 
            nGridX = nGrid[0]
            nGridY = nGrid[1]
            nGridZ = nGrid[2]
            
            # Length (in points) of the grid along the different dimension:
            xlength = axisLength[0]
            ylength = axisLength[1]
            zlength = axisLength[2]

            # Computation of the bin edges:
            binEdges = np.empty([nGridX, nGridY, nGridZ, nBins+1])
        
            for iX in range(nGridX):
                for iY in range(nGridY):
                    for iZ in range(nGridZ):
                        gridZone = image[ xlength[iX] : xlength[iX+1], \
                                          ylength[iY] : ylength[iY+1], \
                                          zlength[iZ] : zlength[iZ+1]]
                        
                        # we transform the 3D matrix into a vector:
                        arrayBig = gridZone.flatten()
                        
                        # lenght of the non-zero array:
                        arrayLenght = len(arrayBig)
                        
                        # We sort the array:
                        arraySort = np.sort(arrayBig)
                        
                        # Number of points per bins:
                        nPtsBins = int(round(arrayLenght / (nBins)))
                        
                        binEdges[iX, iY, iZ, 0] = 0

                        if arraySort[nPtsBins] == 0:
                            
                            # We look for the first non-zero element of the arrayBig:
                            minIndex = np.argmin(arraySort[::-1])    
                            non0Index = arrayLenght - (minIndex)
                            
                            # Array with non zeros elements:
                            arrayNonZeros = arraySort[non0Index:]
                         
                            binEdges[iX, iY, iZ, 1] =  arraySort[non0Index]
                    
                            # lenght of the non-zero array:
                            arrayLenght = len(arrayNonZeros)
                            
                            # Number of points per bins:
                            nPtsBins = int(round(arrayLenght / (nBins-1)))      
                                
                            for i in range(1, nBins):
                                if i != nBins-1:
                                    
                                    binEdges[iX, iY, iZ, i+1] = arrayNonZeros[i*nPtsBins]
                                else:
                                    binEdges[iX, iY, iZ, i+1] = arrayNonZeros[-1]
                                    
                            
#                            print(binEdges[iX, iY, iZ, :])
#                            print(len(binEdges[iX, iY, iZ, :]))
#                            plt.hist(arraySort, bins=binEdges[iX, iY, iZ, :])  
#                    
#                            plt.title("subHistogram with 'auto' bins")
#                            plt.show()
#                            stop

                        else:
                            for i in range(1, nBins+1):
                                if i != nBins:
                                    binEdges[iX, iY, iZ, i] = arraySort[i*nPtsBins]
                                else:
                                    binEdges[iX, iY, iZ, i] = arraySort[-1]
                                                         
                            
                            
        
#                            plt.hist(arraySort, bins=binEdges[iX, iY, iZ, :])  
#                        
#                            plt.title("subHistogram with 'auto' bins")
#                            plt.show()
#                            stop
                            
                        binEdges[iX, iY, iZ, nBins] *= 1.5 
                    
        
        return binEdges
        
        
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
    def __init__(self, featuresMatrix, label=[], multiLabels=[], \
                 classToLabels=[]):
        """
        Constructuctor of the class Prediction
        """
        
        # 2D feature matrix extracted from the original 3D images:
        self.features = featuresMatrix
        
        # Labels of the training dataset:
        self.label = label

        # Multilabels in case of multiple classes (classification):
        self.multiLabels = multiLabels
        
        # Matrix defining the equivalence between the class number and the 
        # multiple labels (gender, age, health):
        self.classToLabels = classToLabels

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
        
        
    def supportVectorMachine(self, regularizerCoeff, typeSVM="Normal", kernelType="rbf"):
        
        featuresMatrix = self.features
        label = self.label
        
        # Type of kernel:
        # Linear kernel
        if kernelType in ["linear", "Linear", "None"]: 
            kernelType = "linear"
        # n-th order Polynomial kernel     
        elif kernelType[0:4] in ["poly", "Poly"]: 
            kernelType = "poly"
            # Order of the polynomial kernel:
            if len(kernelType) == 4:
                polyOrder = 2
            else:
                polyOrder = int(kernelType[4:])
        # Radial basis function (aka homogeneous) kernel    
        elif kernelType in ["rbf", "Radial Basis Function", "RBF", "Homogeneous"]:
            kernelType = "rbf"
        # Sigmoïd kernel    
        elif kernelType in ["sigmoid", "Sigmoid"]:
            kernelType = "sigmoid"
        
        # Type of support vector machine algorithm used for classification:
        if typeSVM == "Normal":
            clf = svm.SVC(C=regularizerCoeff, kernel=kernelType,degree=polyOrder)
        elif typeSVM == "nu":
            clf = svm.NuSVC(nu=regularizerCoeff, kernel=kernelType,degree=polyOrder)
            
        # We train the chosen SVM algorithm on our (training) feature matrix
        clf.fit(featuresMatrix, label)
        
        # We get the parameters of the SVM model:
        parameters = clf.get_params
        
        # Details about the SVM classification:
        # Average distance between the  points and the hyperplane:
        distance2Hyperplane = np.mean(clf.decision_function(featuresMatrix))
        
        return parameters
        
        
    def buildClassifier( self, featureTraining, label, methodDic=[], method = "LASSO"):
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
        
        regression = False
        
        if method == "Linear regression":
            clf = linear_model.LinearRegression(copy_X=True, n_jobs=-1, \
                                                normalize=True)
            regression = True
        elif method == "LASSO":
            clf = linear_model.Lasso(alpha=0.1, copy_X=True, \
                                     normalize=True, tol=0.0001)
            regression = True
        elif method == "Ridge":
            clf = linear_model.Ridge(alpha=0.5, copy_X=True, \
                                     normalize=True, solver='lsqr', tol=0.001)
            regression = True
        elif method == "RidgeCV": #ridge cross validation
            clf = linear_model.RidgeCV( alphas=[0.0, 0.1, 10.0], cv=None, \
                                       fit_intercept=True, normalize=True )
            regression = True

        elif method == "SVM": 
            clf = svm.SVC(**methodDic)
        elif method == "SVC":
            if methodDic == []:
                 clf = SVC(C=1.0, cache_size=200, decision_function_shape='ovr', kernel='rbf', \
                      probability=True, shrinking=True, verbose=False)
            else:
                clf = SVC(**methodDic)
        elif method == "Random Forest":
            clf = RandomForestClassifier(**methodDic)
        elif method == "AdaBoost":
            clf = AdaBoostClassifier(**methodDic)
        elif method == "Bagging":
            clf = BaggingClassifier(**methodDic)
        elif method == "Gradient Boosting":
            clf = GradientBoostingClassifier(**methodDic)
        elif method == "Multi SVM":
            clf = OneVsRestClassifier(svm.SVC(**methodDic))
        elif method == "Multi Random Forest":
            clf = OneVsRestClassifier(RandomForestClassifier(**methodDic))
        elif method == "Multi Output RF":
            clf = MultiOutputClassifier(RandomForestClassifier(**methodDic), n_jobs=-1)
        elif method == "RandomForestClassifier":
            clf = RandomForestClassifier(n_estimators=100, criterion='gini',\
                     bootstrap=True, oob_score=True, n_jobs=-1)
        elif method == "GaussianNB":
            clf = GaussianNB()
        elif method == "GaussianNB_isotonic":
            clf = CalibratedClassifierCV(GaussianNB(), cv=10, method='isotonic')
        elif method == "GaussianNB_sigmoid":
            clf = CalibratedClassifierCV(GaussianNB(), cv=10, method='sigmoid')
        elif method == "MLPClassifier":
            clf = MLPClassifier(alpha=0.001, hidden_layer_sizes=500, activation='logistic', solver='adam', shuffle=True)
        elif method == "KNeighborsClassifier":
            clf =  KNeighborsClassifier(n_neighbors = 2, algorithm='auto', weights='uniform', n_jobs=-1)
        elif method == "GaussianProcess":
            clf = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True, max_iter_predict=200, n_jobs=-1)
        elif method == "AdaBoostClassifier":
            clf = AdaBoostClassifier(n_estimators=100)
        elif method == "VotingClassifier":
            clf = VotingClassifier(estimators=[('lr', LogisticRegression(random_state=1)), \
                                               ('rf', RandomForestClassifier(random_state=1)), \
                                               ('gnb', CalibratedClassifierCV(GaussianNB(), cv=10, method='isotonic'))],\
                                               voting='soft', n_jobs=-1)

        # We train the chosen SVM algorithm on our (training) feature matrix
        clf.fit( featureTraining, label )
        
        if (not regression):
            return clf
        # Use the function coef_ from the sklearn module to compute the 
        # parameters of our model:
        parameters = clf.coef_
        parameters = parameters.reshape(clf.coef_.size, 1)
        parameters = np.append( clf.intercept_, parameters) # to add the bias
        
        return clf, parameters
        
            
    def predict(self, features, method, parameters=[], labelValidation=[], classifier=[]):
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

        if method in ["Random Forest", "AdaBoost", "Bagging", "Gradient Boosting"]:
            predictedData = classifier.predict_proba( features )[:,1]
            #print(predictedData)
            
        elif method == "SVM":
            predictions = classifier.predict( features )
            predictedData = np.zeros(len(predictions))
            
            # We change the values predicted by the support vector machine 
            # classifier: 0-->0.1 and 1-->0.9. This allows reducing the 
            # logistic score:
            for i in range (len(predictions)):
                if predictions[i] == 0:
                    predictedData[i] = 0.1  
                else:
                    predictedData[i] = 0.9

        elif method in ["Multi SVM", "Multi Random Forest"]:
            predictedData = classifier.predict( features )

        else:
            # The first column of the feature matrix should be the bias:
            bias = np.ones([nbSamples,1])
            features = np.concatenate((bias, features), axis=1)
        
            # Prediction of the model for the given dataset:
            predictedData = features.dot( parameters )
        
        # Computation of the error of the predicted data:
        if len(labelValidation) > 0:
            if method in ["Random Forest", "SVM", "AdaBoost", "Bagging", "Gradient Boosting"]:
                error = log_loss(labelValidation, predictedData)
                
            elif method in ["Multi Random Forest", "Multi SVM"]:
                predictionsMultiple = np.zeros([len(labelValidation), 3])
                truePrediction = np.zeros([len(labelValidation), 3])
                for i  in range(len(labelValidation)):
                    predictionsMultiple[i,:] = self.classToLabels[\
                                        predictedData[i],:]
                    
                    # True labels:
                    truePrediction[i,:] = self.classToLabels[\
                                        labelValidation[i],:]

                
                # Score function:
                error = np.mean(np.abs(predictionsMultiple - truePrediction))
                 
                
            else:
                error = (np.mean((predictedData - labelValidation)**2)) 
  
            return predictedData, error
        else:
            return predictedData
        
                    
    def crossValidation(self, methodDic, nFold=10, typeCV="random", \
                        methodList=["SVM"], stepSize = 0.01):
        """ To compute the model parameters through cros-validation and 
        estimate the best weights comination of our models if multiple models
        are selected:
        
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
        
        # Number of different methods used for training the model:
        nModel = len(methodList)

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
        scoreArray = np.zeros([nFold])
        score = np.zeros([nModel])
        
        # Matrix containing all the predictions:
        predictions = np.empty([sampleTrainCV])
        
        # List containing the predicted data computed by all the models 
        # on the n-th fold:
        #predictionFold = []

        # List containing all the predictions:
        predictMatrix = np.zeros([sampleTrainCV, nModel])
        
        # Number of steps for which we try to find the best weigth value:
        nSteps = 1 + 1 / stepSize
        
        # range of values where we try to find the best weigth value:
        rangeVector = np.linspace(0.5, 1, nSteps, endpoint=True)

        # Weigths of the different models used for computing the combined model:
        weightModel = np.ones([nModel])
        
        # True labels:
        labelTrue = label[:sampleTrainCV]
        
        # Computation of the ranking of the different models computed by the 
        # different techniques:
        for n, inputMethod in enumerate(methodDic):
            for i in range(nFold):
                
                # We divide the all training dataset into K buckets (or folds):
                indexValid = np.arange(i*samplePerFoldTrain, i*samplePerFoldTrain + sampleValid )
                indexTrain = np.delete(indices, indexValid)
                
                # We compute the model parameters of the chosen technique:
                clf = self.buildClassifier(featuresMatrix[indexTrain, :], \
                                label[indexTrain], methodDic=inputMethod, method=methodList[n])
                
               
                # We predict the data with the computed parameters
                predictedData, scoreArray[i] = self.predict( \
                    features=featuresMatrix[indexValid, :], method=methodList[n], \
                    parameters=[], labelValidation=label[indexValid], classifier=clf)
                
                # We save the predicted data in the appropriate matrix:
                predictions[i*samplePerFoldTrain : (i+1)*samplePerFoldTrain] \
                            = predictedData[:samplePerFoldTrain]

            
            # Variance of the model:
            varianceModel = np.round(np.var(scoreArray),4)

            if methodList[0][:5] == "Multi":
                # Score array:
                # 1) Final score (Hamming Loss):
                # 2) Gender score:
                # 3) Age score:
                # 4) Health score:
                score = - np.ones(4)
                
                predictions = predictions.astype(int)
                # The class number is transformed into the triplet  
                # (gender, age, health status):
                predictionsMultiple = np.zeros([sampleTrainCV, 3])
                for i  in range(sampleTrainCV):
                    predictionsMultiple[i,:] = self.classToLabels[\
                                        predictions[i],:]
                    
                # True labels:
                truePrediction = self.multiLabels[:sampleTrainCV]
                
                # Score function:
                score[0] = np.mean(np.abs(predictionsMultiple - truePrediction))
                
                # Score for the gender prediction:
                for i in range(3):
                    score[i+1] = np.mean(np.abs(predictionsMultiple[:,i] - \
                        truePrediction[:,i]))
                score = np.round(score,4)
                    
            else:
                # Score:
                score[n] = log_loss(labelTrue, predictions)
                                
                # Best weighted prediction:
                predictionBest = predictions
                
                # New prediction obtained by weighting the model:
                predictNew = predictions
    
                # Best score achieved:
                scoreBest = score[n]
                
                for k in rangeVector:
                    predictNew = k*predictionBest
                    error = log_loss(labelTrue, predictNew)
                    if error < scoreBest:
                        scoreBest = error
                        weightModel[n] = k
                
                # Best score of the model n:
                score[n] = round(scoreBest,4)

            # The predictions of the model n are put into the matrix containing 
            # all the predictions given by all the models:
            predictMatrix[:,n] = weightModel[n] * predictions
                
            # We compute the overall mean-squared error:
            #score[n] = round(np.mean(scoreArray),4)
              
        
        # We sort the different models with respect to their scores (from 
        # the best to the worst):
        
        rankIndex = np.argsort(score)
        print( "rank index is:")
        print( rankIndex )
      
        methodDicOrd = []
        methodListOrd = []
    
        # If there is more than one model, we compute the score for each model:
        if nModel > 1:
            score = score[rankIndex]
            weightModel = weightModel[rankIndex]
            for i in range(nModel):
                methodDicOrd.append(methodDic[rankIndex[i]])
                methodListOrd.append(methodList[rankIndex[i]])
            predictMatrix = predictMatrix[:, rankIndex] 
        
            methodDic = methodDicOrd
            methodList = methodListOrd
        
        # Print the ranking of the different methods and their scores:
        if nModel == 1:
            print("{} method performs a score of {}".format(methodList[0], score[0]))
             
        # ENSEMBLE SELECTION: Computation of the weigths assigned to each model
        # in order to determine the best averaged model:

        if nModel > 1:

            # range of values where we try to find the best weigth value:
            rangeVector = np.linspace(0, 1, nSteps, endpoint=True)
            
            # Best weighted prediction:
            predictionBest = predictMatrix[:,0]
            
            # New prediction obtained by combining the different weighted models:
            predictNew = predictMatrix[:,0]

            # Best score achieved:
            scoreBest = score[0]
            
            # We check each model in the order of their score:
            for n in range(1,nModel):
                # With a given step, we try to find the weight to assign to 
                # model in order to obtain a better score:
                for i in rangeVector:
                    # The weight i is tried:
                    predictNew = (1-i)*predictionBest + i*predictMatrix[:,n]
                    # The error with the weight i is computed:
                    error = log_loss(labelTrue, predictNew)
                    
                    # if we obtain a better (=lower) score we save the weight 
                    # to assign to our models:
                    if error < scoreBest: 
                        scoreBest = error
                        bestIndex = i
                
                # The best weight is then saved:
                for k in range(n):
                    weightModel[k] = (1-bestIndex)*weightModel[k]   
                weightModel[n] = bestIndex * weightModel[n]
                
                # We compute the prediction with the best weighted combinations 
                # of our models:
                predictionBest = np.zeros([sampleTrainCV])
                for k in range(n+1):
                    predictionBest += predictMatrix[:,k] * weightModel[k]
            # We compute the score with the best weighted combinations 
            # of our models:
            scoreBest = round(scoreBest,4)
        
        if methodList[0][:5] != "Multi":   
            # we sort the label (ascending order) and the predicted data:
            indexSort = np.argsort(labelTrue)
            labelSort = np.array(labelTrue[indexSort])
            
            if nModel == 1:
                predictedDataSort = np.array(predictions[indexSort])
            else:
                predictedDataSort = np.array(predictionBest[indexSort])
            
            # number of sick people in the dataset:
            nSick = np.argmax(labelSort) 
            
            # Prediction average for the class 0 (sick):
            predictSickAvg = round(np.mean(predictedDataSort[:nSick-1]),2)
            
            # Prediction average for the class 1 (healthy):
            predictHealthyAvg = round(np.mean(predictedDataSort[nSick:]),2)
            
            print("\nMean prediction:\nfor sick people: {}\nfor healthy people:{}".format(predictSickAvg, predictHealthyAvg))
            if nModel > 1:
                print("\nRanking of the models:")
                for k in range(nModel):
                    print("{}) {}: {}".format(1+k, methodList[k], score[k]))
                
                print("--> Ensemble selection: {}".format(scoreBest))
     
            noPlot=False
            if noPlot:
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
                plt.ylabel("Health condition")
        
        if nModel == 1:
            return score, varianceModel
        else:
            return scoreBest, weightModel

    def ensembleSelection(self, methodDic, methodList, Ratio=0.8,\
                          typeDataset="random", stepSize=0.001):

        """ To compute the model parameters through cros-validation and 
        estimate the best weights comination of our models: 
        
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
        
        # Number of different methods used for training the model:
        nModel = len(methodList) 
        
        
        # We shuffle the indices of the feature matrix in the first dimension 
        # (number of samples)
        if typeDataset == "random":
            # Array with the indices of the sample in the dataset
            indices = np.arange(nSamples)
            
            # We shuffle the indices
            indicesShuffled = np.random.shuffle(indices)
            featuresMatrix = featuresMatrix[indicesShuffled, :][0,:,:]
            label = label[indicesShuffled][0,:]

        # Array with the indices of the sample in the dataset:
        indices = np.arange(nSamples)
        
        # Number of samples used for training over all the cross validation 
        # process
        sampleTrain = int(np.floor(Ratio * nSamples))

        # Number of samples used for validation
        sampleValid = nSamples - sampleTrain
        
        # Creation of the dictionary containing the indices of the samples used
        # for training for each bucket
        indexTrain = np.arange(sampleTrain)
       
        # Creation of the dictionary containing the indices of the samples used
        # for validation for each bucket
        indexValid = np.arange(sampleTrain, nSamples)   
        
        # Creation of the array containing the MSE error for each cross 
        # validation
        score = np.zeros([nModel])        
        
        # Matrix containing all the predictions:
        predictions = np.empty([sampleValid, nModel])
        
        # Number of steps for which we try to find the best weigth value:
        nSteps = 1 + 1 / stepSize
        
        # range of values where we try to find the best weigth value:
        rangeVector = np.linspace(0.5, 1, nSteps, endpoint=True)
        
        # Weigths of the different models used for computing the combined model:
        weightModel = np.ones([nModel])

        # True labels:
        labelTrue = label[indexValid]

        # List containing the classifiers of each model:
        clf =[]
        
        # Computation of the ranking of the different models computed by the 
        # different techniques:
        for n, inputMethod in enumerate(methodDic):
                                
            # We compute the model parameters of the chosen technique:
            clf.append( self.buildClassifier(featuresMatrix[indexTrain, :], \
                            label[indexTrain], methodDic=inputMethod, method=methodList[n]))
           
            # We predict the data with the computed parameters
            predictions[:,n], score[n] = self.predict( \
                features=featuresMatrix[indexValid, :], method=methodList[n], \
                parameters=[], labelValidation=labelTrue, classifier=clf[n])
 
            # Best weighted prediction:
            predictionBest = predictions[:,n]
            
            # New prediction obtained by weighting the model:
            predictNew = predictions[:,n]

            # Best score achieved:
            scoreBest = score[n]
            
            for k in rangeVector:
                predictNew = k*predictionBest
                error = log_loss(labelTrue, predictNew)
                if error < scoreBest:
                    scoreBest = error
                    weightModel[n] = k
            
            # Best score of the model n:
            score[n] = round(scoreBest,4)

            # The predictions of the model n are put into the matrix containing 
            # all the predictions given by all the models:
            predictions[:,n] *= weightModel[n]                 
        
        # We sort the different models with respect to their scores (from 
        # the best to the worst):
        rankIndex = np.argsort(score)
         
        methodDicOrd = []
        methodListOrd = []

        if nModel > 1:
            score = score[rankIndex]
            weightModel = weightModel[rankIndex]
            for i in range(nModel):
                methodDicOrd.append(methodDic[rankIndex[i]])
                methodListOrd.append(methodList[rankIndex[i]])
            predictions = predictions[:, rankIndex] 
        
        methodDic = methodDicOrd
        methodList = methodListOrd
        
        print(weightModel)
        
        # Print the ranking of the different methods and their scores:
        if nModel == 1:
            print("{} method performs a score of {}".format(methodList[0], score[0]))
             
        # ENSEMBLE SELECTION: Computation of the weigths assigned to each model
        # in order to determine the best averaged model:

        if nModel > 1:

            # range of values where we try to find the best weigth value:
            rangeVector = np.linspace(0, 1, nSteps, endpoint=True)
            
            # Best weighted prediction:
            predictionBest = predictions[:,0]
            
            # New prediction obtained by combining the different weighted models:
            predictNew = predictions[:,0]

            # Best score achieved:
            scoreBest = score[0]
            bestIndex = 0
            
            for n in range(1,nModel):
                
                for i in rangeVector:
                    predictNew = (1-i)*predictionBest + i*predictions[:,n]
                    error = log_loss(labelTrue, predictNew)
                    if error < scoreBest:
                        
                        scoreBest = error
                        bestIndex = i
                
                for k in range(n):
                    
                    weightModel[k] = (1-bestIndex)*weightModel[k]
                   
                weightModel[n] = bestIndex * weightModel[n]
                
                predictionBest = np.zeros([sampleValid])
                for k in range(n+1):
                    
                    predictionBest += predictions[:,k] * weightModel[k]
            
            scoreBest = round(scoreBest,4)
            
        # we sort the label (ascending order) and the predicted data:
        indexSort = np.argsort(labelTrue)
        labelSort = np.array(labelTrue[indexSort])
        
        if nModel == 1:
            predictedDataSort = np.array(predictions[indexSort])
        else:
            predictedDataSort = np.array(predictionBest[indexSort])
        
        # number of sick people in the dataset:
        nSick = np.argmax(labelSort) 
        
        # Prediction average for the class 0 (sick):
        predictSickAvg = round(np.mean(predictedDataSort[:nSick-1]),2)
        
        # Prediction average for the class 1 (healthy):
        predictHealthyAvg = round(np.mean(predictedDataSort[nSick:]),2)
        
        print("\nMean prediction:\nfor sick people: {}\nfor healthy people:{}".format(predictSickAvg, predictHealthyAvg))
        if nModel > 1:
            print("\nRanking of the models:")
            for k in range(nModel):
                print("{}) {}: {} ({}%)".format(1+k, methodList[k], score[k],\
                        round(100*weightModel[k])))
            
            print("--> Ensemble selection: {}".format(scoreBest))
            
        # Plot the predicted data and the true data:
        plt.figure(100)
        
         # X-axis:
        x = np.linspace(1, sampleValid, sampleValid)
            
        # Plot of the predicted labels:
        plt.plot(x, predictedDataSort, color="blue", linewidth=1, \
                 linestyle='--', marker='o')
        
        # Plot of the true labels:
        plt.plot(x, labelSort, color="red", linewidth=1, \
                 linestyle='--', marker='o')
       
        plt.title("Validation of the model")
        plt.xlabel("Patient number")
        plt.ylabel("Health condition")
        
        return clf, weightModel, scoreBest

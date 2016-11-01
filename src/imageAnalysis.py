# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:55:03 2016

@author: valentin
"""
import numpy as np
import matplotlib.pyplot as plt

class Image2D:
    def __init__(self, array2D):
        self.array = array2D
        self.size = array2D.shape
        self.max = self.array.max()
        self.min = self.array.min()
        
        # Size of the 2D matrix:
        self.size2D = self.array.shape
        (self.sizeX,self.sizeY) = self.size2D
        
        # Central positions along x, y, z:
        self.X0 = int(float(self.sizeX)/2)
        self.Y0 = int(float(self.sizeY)/2)
        
    def grid2D(self, xlength=10, ylength=10):
        
        xlength = round(xlength*self.sizeX/100.0)
        ylength = round(ylength*self.sizeX/100.0)
        
        nGridX = np.ceil(100*self.sizeX / xlength) 
        nGridY = np.ceil(100*self.sizeY / ylength) 

        # 2D average processed wrt to the grid:
        gridZones = np.empty([nGridX*nGridY, 2])
        
        for i in range(gridZones-1):
            gridZones[i, :] = [i*xlength % nGridX, i*ylength % nGridY]
        
        return gridZones
        
    

###############################################################################
class ImageProperties:
    
    def __init__(self, array3D):
        self.array = array3D
        self.size = array3D.shape
        self.max = np.amax(self.array)
        self.min = np.amin(self.array)
        
        # Size of the 3D matrix:
        self.size3D = array3D.shape
        (self.sizeX,self.sizeY,self.sizeZ) = self.size3D
        
        # Central positions along x, y, z:
        self.X0 = int(float(self.sizeX)/2)
        self.Y0 = int(float(self.sizeY)/2)
        self.Z0 = int(float(self.sizeZ)/2)
        
    def __repr__(self):
        #dataType = self.array.dtype
        return "This object is a 3D array from MRI images. It contains {}*{}*{} elements \
        ".format(self.X0, self.Y0, self.Z0) 
        
    def compare(self, image2):
        #array1 = self.array
        #array2 = image2.array
        
        # Difference between the 2 images:
        #arrayDiff = array1 - array2
        
        # Creation of the class object ImageProperties from the difference 
        # between the 2 images:
        #imageDiff = ImageProperties(arrayDiff)
        
        # We call the function toPlot to plot some 2D images on two different 
        # plots:
        self.toPlot(1)
        image2.toPlot(2)
        #imageDiff.toPlot(3)
        
        
    def to2D(self, type):
        if type=="slice":
            imageXY = self.array [:,:,self.Z0]
            imageXZ = self.array [:,self.Y0,:]
            imageYZ = self.array [self.X0,:,:]
        elif type=="sum" or type=="average":
            imageXY = self.array .sum(axis=2)
            imageXZ = self.array .sum(axis=1)
            imageYZ = self.array .sum(axis=0)
        else:
            imageXY = imageXZ = imageYZ = 0
            
        return imageXY, imageXZ, imageYZ
            
        
    def toPlot(self, fIndex = 1):
        plt.figure(fIndex)

        # Load the 2D slices using the function to2D:
        imageX_Y_Z0, imageX_Y0_Z, imageX0_Y_Z = self.to2D(type="slice")
        
        # Load the 2D sum image using the function to2D:
        imageXY, imageXZ, imageYZ = self.to2D(type="sum")

        # Plot of slices along the 3D:
        plt.subplot(2, 3, 1)
        plt.imshow(imageX_Y_Z0)
        plt.title("image(X,Y,Z0)")
        plt.xlabel("Y")
        plt.ylabel("X")
        
        plt.subplot(2, 3, 2)
        plt.imshow(imageX_Y0_Z)
        plt.title("image(X,Y0,Z)")
        plt.xlabel("Z")
        plt.ylabel("X")
        
        plt.subplot(2, 3, 3)
        plt.imshow(imageX0_Y_Z)
        plt.title("image(X0,Y,Z)")
        plt.xlabel("Z")
        plt.ylabel("Y")
        
        # Plot of 2D images resulting from the sum over 1D:
        plt.subplot(2, 3, 4)
        plt.imshow(imageXY)
        plt.title("image(X,Y)")
        plt.xlabel("Y")
        plt.ylabel("X")
        
        plt.subplot(2, 3, 5)
        plt.imshow(imageXZ)
        plt.title("image(X,Z)")
        plt.xlabel("Z")
        plt.ylabel("X")
        
        plt.subplot(2, 3, 6)
        plt.imshow(imageYZ)
        plt.title("image(Y,Z)")
        plt.xlabel("Z")
        plt.ylabel("Y")
###############################################################################
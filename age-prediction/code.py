import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import glob

from sklearn.feature_selection import f_classif, SelectKBest
from nibabel.testing import data_path
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR

#one mean for each image
meanFeature = []

#code show_slices from: http://nipy.org/nibabel/coordinate_systems.html#introducing-someone
def show_slices( slices ):
    """ Function to display row of image slices """
    fig, axes = plt.subplots( 1, len( slices ) )
    for i, slice in enumerate( slices ):
        axes[i].imshow( slice.T, cmap="gray", origin="lower" )

#------------------------------------------------------------------------------------------

def calculate_mean( path ):
    files = glob.glob( path )
    files.sort()
    #dimension of the data is 176 208 176
    xShape = 176
    yShape = 208
    zShape = 176
    meanFeature = np.zeros( ( len( files ), 1 ) )
    # for all files in the folder
    for i in range( len( files ) ):
        #load file
        img = nib.load( files[i] )
        print( "Image ", i )
        #get the data (one file)
        X_data = img.get_data()
        
        #calculate mean for all image
        sum_intensities = 0
        voxels = 0
        for x in range( xShape ):
            for y in range( yShape ):
                for z in range( zShape ):
                    sum_intensities += X_data[x,y,z,0]
                    if X_data[x,y,z,0] > 0:
                        voxels += 1
        meanFeature.append( sum_intensities/voxels )
        print ( meanFeature[i] )
    return meanFeature

#------------------------------------------------------------------------------------------
def calculate_8x8_mean( path ):
    files = glob.glob( path )
    files.sort()
    #dimension of the data is 176 208 176
    xShape = 22 #176/8
    yShape = 26 #208/8
    zShape = 22 #176/8
    
    meanFeature = np.zeros( ( len( files ), 22 * 26 * 22 ) )
    # for all files in the folder
    for i in range( len( files ) ):
        #load file
        img = nib.load( files[i] )
        print( "Image ", i )
        #get the data (one file)
        X_data = img.get_data()
        X_data_float64 = X_data.astype(np.float64)
        
        #calculate mean for all image
        sum_intensities = 0
        for x in range( xShape ):
            for y in range( yShape ):
                for z in range( zShape ):
                    image = X_data_float64[(8 * x):(8 * x + 8), (8 * y):(8 * y + 8), (8 * z):(8 * z + 8)]
                    img_ravel = image.ravel()
                    meanFeature[i, 26 * 22 * x + 22 * y + z] = np.mean( img_ravel )
    return meanFeature
#---------------------------------------------------------------------

def write_feature( path, feature ):
    fileIO = open( path, 'w' )
    for i in range( len( feature ) ):
        fileIO.write( str( feature[i] ) )
    fileIO.close()

#----------------------------------------------------------------------

#training data
path = "data/set_train/*"

#one mean for all the volume
#if not os.path.isfile( 'completeMeanTraining.txt' ):
#    meanFeature = calculate_mean( path )
#    write_feature( 'completeMeanTraining.txt', meanFeature )
#else:
#    fileIO = open( 'completeMeanTraining.txt', 'r' )
#    meanFeature = fileIO.readlines()
#    fileIO.close()

#a = np.array( meanFeature ).astype( np.float )
#a = a.reshape( 278, 1 )

#one mean for each 8x8 volume
if not os.path.isfile( 'data/8x8MeanTraining.txt' ):
    meanFeature = calculate_8x8_mean( path )
    write_feature( 'data/8x8MeanTraining.txt', meanFeature )
else:
    fileIO = open( 'data/8x8MeanTraining.txt', 'r' )
    meanFeature = fileIO.readlines()
    fileIO.close()

a = np.array( meanFeature ).astype( np.float )
a = a.reshape( 278, 22 * 26 * 22 )

print( "training data - features - OK" )

fileIO = open( 'data/targets.csv', 'r' )
y = fileIO.readlines()
fileIO.close()

print( "training data - answers - OK" )

b = np.array( y ).astype( np.float )
b = b.reshape( 278, 1 )

print(a.size)

#divide the training data for training and validation
x_train, x_test, y_train, y_test = train_test_split( a, b, test_size = 0.33 )
#print(x_train.size)
#print(y_train.size)
#print(x_test.size)
#print(y_test.size)

print( "data separated for training and validation" )

#regression = linear_model.RidgeCV( alphas=[0.1, 1.0, 10.0] )
regression = SVR(kernel='rbf', C=1e3, gamma=0.1)
regression.fit( x_train, y_train )       
#RidgeCV( alphas=[0.1, 1.0, 10.0], cv=None, fit_intercept=True, scoring=None, normalize=False )
print( "RidgeCV trained" )

#The mean square error
print( mean_squared_error( y_test, ( regression.predict( x_test ) ) ) )

#test data
path = "data/set_test/*"

#one mean for all the volume
#if not os.path.isfile( 'completeMeanTest.txt' ):
#    meanFeature = calculate_mean( path )
#    write_feature( 'completeMeanTest.txt', meanFeature )
#else:
#    fileIO = open( 'completeMeanTest.txt', 'r' )
#    meanFeature = fileIO.readlines()
#    fileIO.close()

#one mean for each 8x8 volume
if not os.path.isfile( 'data/8x8MeanTest.txt' ):
    meanFeature = calculate_8x8_mean( path )
    write_feature( 'data/8x8MeanTest.txt', meanFeature )
else:
    fileIO = open( 'data/8x8MeanTest.txt', 'r' )
    meanFeature = fileIO.readlines()
    fileIO.close()
    
print( "test data - features - OK" )

#writing answer
fileIO = open( 'data/submission.csv','w' )
fileIO.write( 'ID,Prediction\n' )
y_ = regression.predict( meanFeature )
for i in range( len( y_ ) ):
    fileIO.write( str(i+1) + ',' + str(y_[i]).strip('[]') + '\n' )
fileIO.close()



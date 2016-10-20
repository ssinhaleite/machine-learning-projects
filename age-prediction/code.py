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

#one mean for each image
completeMean = []

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
    completeMean = []
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
        completeMean.append( sum_intensities/voxels )
        print ( completeMean[i] )
    return completeMean

#---------------------------------------------------------------------

def write_feature( path, feature ):
    fileIO = open( path, 'w' )
    for i in range( len( feature ) ):
        fileIO.write( str( feature[i] ) )
    fileIO.close()

#----------------------------------------------------------------------

#training data
path = "data/set_train/*"
if not os.path.isfile( 'completeMeanTraining.txt' ):
    completeMean = calculate_mean( path )
    write_feature( 'completeMeanTraining.txt', completeMean )
else:
    fileIO = open( 'completeMeanTraining.txt', 'r' )
    completeMean = fileIO.readlines()
    fileIO.close()
print( "training data - features - OK" )

fileIO = open( 'targets.csv', 'r' )
y = fileIO.readlines()
fileIO.close()

print( "training data - answers - OK" )

a = np.array( completeMean ).astype( np.float )
a = a.reshape( 278, 1 )
b = np.array( y ).astype( np.float )
b = b.reshape( 278, 1 )

#divide the training data for training and validation
x_train, x_test, y_train, y_test = train_test_split( a, b, test_size = 0.33 )
#print(x_train.size)
#print(y_train.size)
#print(x_test.size)
#print(y_test.size)

print( "data separated for training and validation" )

regression = linear_model.RidgeCV( alphas=[0.1, 1.0, 10.0] )
regression.fit( x_train, y_train )       
RidgeCV( alphas=[0.1, 1.0, 10.0], cv=None, fit_intercept=True, scoring=None, normalize=False )
print( "RidgeCV trained" )

#The mean square error
print( mean_squared_error( y_test, ( regression.predict( x_test ) ) ) )

#test data
path = "data/set_test/*"
if not os.path.isfile( 'completeMeanTest.txt' ):
    calculate_mean( path )
    write_feature( 'completeMeanTest.txt', completeMean )
else:
    fileIO = open( 'completeMeanTest.txt', 'r' )
    completeMean = fileIO.readlines()
    fileIO.close()
print( "test data - features - OK" )

#writing answer
fileIO = open( './submission.csv','w' )
fileIO.write( 'ID,Prediction\n' )
for i in range( len( completeMean ) ):
    y_ = regression.predict( float( completeMean[i] ) )
    print( y_ )
    fileIO.write( str(i+1) + ',' + str(y_) + '\n' )
fileIO.close()



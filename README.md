# Machine Learning Projects - 252-0535-00L - Fall 2016

## Team: kareny
### Members:
* Karen Yeressian Negarchi
* Valentin Gallet
* Vanessa Rodrigues Coelho Leite

Project 01
==========

Objective:
----------
- Predicting a person's age from his 3D brain magnetic resonance image (MRI). In order to reduce the volume of the data, some specific features are extracted from the MRI. To achieve the age prediction, a labeled dataset is used for training our prediction model using different kind of linear regression techniques (Lasso, Ridge). The accuracy of our model and our feature selection are then estimated by cross-validation. Once our better model selected, we applied it to the test dataset (non-labeled) to predict the patient's age.  


About the code:
---------------

### General:
This code was designed in order to be used as a structure for the three projects in this course. It is carefully commented and organized.
Our code is divided into three files as described below:
- **imageAnalysis.py**: contains definition of MRI as an image. Used for plot the MRI and to compare two images.
Example of use:

```
image_1 = ia.ImageProperties(dataset[0]**1.)
image_2 = ia.ImageProperties(dataset[1]**1.)

image_1.compare(image_2)
```

- **MachineLearningAlgorithms.py**: contains two classes - Features and Prediction.
In the Feature class you can find our implementation for feature extraction. We extract features from 2D or 3D subsamples of the whole 3D MRI. We implement a `grid` and a `threshold` functions. The `grid` function divide the space into small volumes (3D)or areas (2D) on which some mathematical opérations are processed (mean, variance, maximum value...). The `threshold`function counts the number of voxels being in a specific intensity range. However, this function has not been carefully tested yet and we strongly recommand not to use it before a new version.

|			 | 2D Features | 3D Features |
|:-----:|:-----------:|:-----------:|
| Mean | x| x|
| Max | x| x|
| Min | x| x|
| Variance | x| x|
| Covariance | x| x|
| Sum | x| x|
| energy | x|	|
| contrast | x|	|
| dissimilarity | x|	|
| homogeneity | x|	| |

In the Prediction class, a function (buildclassifier) is defined to perform linear regression using technique like Ridge or Lasso. We first implemented our own version of Ridge but since it gives exactly the same results as the one defined  in the machine learning library sklearn but in a longer time. This is why, we have decided to use this library. We also coded a function (crossvalidation) to perform a cross-validation in order to validate our model and feature selection.

Linear Regression techniques usables within the buildclassifier function:

1. Basic linear regression (without regularization)
2. Lasso
3. Ridge regression
4. RidgeCV (cross validation)
5. SVR-Linear (Support Vector Regression with Linear kernel)


- **predict_final.py**: This script contains the main structure of our code, i. e., the whole workflow. Among other things, it allows doing the following actions:
1. It loads the training data and its labels, 
2. It extracts the right features, 
3. It finds the parameters of our model,
4. It estimates the accuracy of our model and feature selection through a cross-validation stage,
5. It load the test data, extracts the features and gets the predictions 
6. and write the result in a output file.

### Dependencies:
* python 3.5
* numpy
* nibabel
* importlib
* matplotlib
* sklearn
* skimage

### Configure:

For use this code you need to have define three variables:
* `path`: path where your data (training and test) are saved.
* `strDataSet`: name of the folder where you data is saved. We assume the folder's name for training is `set_train` and the folder's name for test is `set_test`
* `strName`: name of your nii files (mri images). We assume the name of training data files is `train_x.nii` and for the test data files is `test_x.nii`, where x is the number of the file.

### Usage:
For use our code you just need to run in a cmd command:
```
python predict_final.py
```
After that, our programm will run, printing some informations on terminal and will write in the informed `path` a file named submission.csv. This output file has two columns: the first is the number of the test data and the second is the age prediction. The first line of each output file contains a header( ID,Prediction).

### Approache:
For predict a persons age using the MRI information, we used the follow approache:
* preprocessing: the MRI volumes contains some areas with no information. We identified visually the region that limits our interested areas. This way, we discard the first 14 and last 18 voxels in `x` dimension, the first 12 and the last 15 voxels in `y` dimension and the first 3 and last 20 voxels in `z` dimension.

* feature extraction: for all voxels left we defined a mask (can be either 2D or 3D) and we operate some of the calculation mentioned above. In our test, the best result was achieved using mean and variance with a 15x15x15 mask.

* train model: after extract features we were able to train our model. We experienced various classifiers but we ended choosing the RidgeCV method.

* prediction: once we have the model trained, we just need to load the test data and do the same preprocessing and feature extraction. After that, we get the predictions and writed than in a text file.

### References:
For choose our features and model we read some papers:
```
* Aman Chadha, Sushmit Mallik and Ravdeep Johar, "Comparative Study and Optimization of Feature-Extraction Techniques for Content based Image Retrieval" (International Journal of Computer Applications, 2012)
* Isabelle Guyon and André Elisseeff, "An Introduction to Feature Extraction" ("Feature Extraction: Foundations and Applications", Springer Berlin Heidelberg, 2006)
* Longfei Su, Lubin Wang and Dewen Hu, "Predicting the Age of Healthy Adults from Structural MRI by Sparse Representation" ("Intelligent Science and Intelligent Data Engineering: Third Sino-foreign-interchange Workshop, IScIDE 2012)
```

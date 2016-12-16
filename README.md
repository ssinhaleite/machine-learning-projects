# Machine Learning Projects - 252-0535-00L - Fall 2016

## Team: kareny / MachineSearching
### Members:
* Karen Yeressian Negarchi
* Valentin Gallet
* Vanessa Rodrigues Coelho Leite

## Dependencies:
* python 3.5
* numpy
* nibabel
* importlib
* matplotlib
* sklearn
* skimage

## Configure:

For using this code you need to define three variables:
* `path`: the path of the folder where your data (training and test) are located.
* `strDataSet`: name of the folder where you data is located. We assume the folder's name for training is `set_train` and the folder's name for test is `set_test`
* `strName`: name of your nii files (3d MRI). We assume the name of training data files is `train_x.nii` and for the test data files is `test_x.nii`, where x is the number of the file.


Project 01
==========

Objective:
----------
Predicting a person's age from his 3D brain magnetic resonance image (MRI). In order to reduce the volume of the data, some specific features are extracted from the MRI. To achieve the age prediction, a labeled dataset is used for training our prediction model using different kind of linear regression techniques (Lasso, Ridge). The accuracy of our model and our feature selection are then estimated by cross-validation. Once our better model selected, we applied it to the test dataset (non-labeled) to predict the patient's age.

Approach:
---------

To predict a persons age using the MRI information, we used the following approach:

* pre-processing: the MRI data contain some areas with no relevent information. We identified visually the region that limits our areas of interest and therefore discarded the first 14 and last 18 voxels in `x` dimension, the first 12 and the last 15 voxels in `y` dimension and the first 3 and last 20 voxels in `z` dimension.

* feature extraction: After having selected the voxels of interest, we divided the 3D MRI into small volumes (3D) or areas (2D) on which some mathematical operations were processed (mean, variance, maximum value...).

* Creation of the model: We train our model on the extracted features. We estimate the accuracy of the model by testing different combination of features and of linear regression techniques. We finally chose the model given the smallest mean-squared error after cross-validation. Inparticular, the best results was achieved using mean and variance with a 15x15x15 grid and the ridge regression.

* prediction: Once our model defined, we applied this latter to the test dataset with the same preprocessing and feature extraction to finally obtain the predictions and written them in a text file.

Evaluation:
-----------
The evaluation metric for this project is Mean-Squared-Error (MSE). The MSE score, a basic measure of fit, represents the average deviation of the predictions from their true values. The MSE metric weights large deviations much heavier than small deviations. Consequently, it is particularly vulnerable to outliers.

Project 02:
===========

Objective:
----------

Classifying brain health status from MR images. In this project we classify a person's cognitive health status only from an MR scan of their brain.

Approach:
---------

Evaluation:
-----------
The evaluation metric for this project is binary Log-Loss (LL), also known as Cross-Entropy-Loss. The Log-Loss punishes wrong predictions (i.e. false positives and false negatives) compared to the true labels.



Project 03:
===========

Objective:
----------

Simultaneously classify a person's gender, age and cognitive health status based on an MR image of their brain. This third project tries to combine the previous efforts (for Project 01 and Project 02) and extends it to a multi-label problem. There are three labels, as mentioned before they are gender(male/female), age(young/old) and cognitive health status(sick/healthy).

For the third project we were not allowed to use targets from Project 01 and Project 02.

Approach:
---------

Evaluation:
-----------

For evaluation, we use the Hamming Loss which measures the accuracy for a multi-label classification task. Assume that `y_i_j` are the ground truth labels, where index `i` is the sample number and `j` is the label number. We have three labels, `j ∈ L:={1,...,3}`(1,2,3 respectively for gender, age, and healthy status). Note that all labels are Boolean in this project. For example, if sample 1 is healthy then `y_0_3 = True`. Given Boolean predictions `y'_i_j`, one can compute the Hamming Loss as

```
HammingLoss(y',y) = 1/|D| * ∑i=0|D|−1 ∑j=1|L| (xor(y'_i_j,y_i_j)/|L|)
```

where

`|D|` is the number of samples (here is the number of test samples, i.e. 138)

`|L|` is the number of labels (here is 3)

`y` is the ground truth (a 138 × 3 matrix)

`y'` is the prediction (a 138 × 3 matrix)


About the code:
===============

General:
--------

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
In the Feature class you can find our implementation for feature extraction. We extract features from 2D or 3D subsamples of the whole 3D MRI. We implement a `grid` and a `threshold` functions. The `grid` function divide the space into small volumes (3D) or areas (2D) on which some mathematical operations are processed (mean, variance, maximum value...). The `threshold`function counts the number of voxels being in a specific intensity range. However, this function has not been carefully tested yet and we strongly recommand not to use it before a new version.

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
1.It loads the training data and its labels, 
2.It extracts the right features, 
3.It finds the parameters of our model,
4.It estimates the accuracy of our model and feature selection through a cross-validation stage,
5.It load the test data, extracts the features and gets the predictions 
6.and write the result in a output file.


Usage:
------
To use our code you just need to configure your project (as explained above) and run in a `cmd` command:

```
python predict_final.py
```

After that, our programm will run, printing some informations on terminal and will write in the informed `path` a file named submission.csv. 

* For prediction: the output file has two columns; the first is the number of the test data and the second is the age prediction. The first line of each output file contains a header (ID, Prediction).
* For classification: the output file has one column; Note that the true labels are either `0` or `1`, but our results are in the interval between 0 and 1, which reflects the uncertainty of our prediction.


References:
===========

To choose our features and models we use the materials given during the lectures and seminares and by reading some papers and books:
```
* Bishop, "Pattern Recognition And Machine Learning" - Springer  2006
* Aman Chadha, Sushmit Mallik and Ravdeep Johar, "Comparative Study and Optimization of Feature-Extraction Techniques for Content based Image Retrieval" (International Journal of Computer Applications, 2012)
* Isabelle Guyon and André Elisseeff, "An Introduction to Feature Extraction" ("Feature Extraction: Foundations and Applications", Springer Berlin Heidelberg, 2006)
* Longfei Su, Lubin Wang and Dewen Hu, "Predicting the Age of Healthy Adults from Structural MRI by Sparse Representation" ("Intelligent Science and Intelligent Data Engineering: Third Sino-foreign-interchange Workshop, IScIDE 2012)
```

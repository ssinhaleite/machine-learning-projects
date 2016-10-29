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
- Predicting a person's age from brain MR images: in this project we apply regression techniques to predict a person's age from a three-dimensional magnetic resonance (MR) image of their brain.


About the code:
---------------

### General:
This code was created as an structure for the three projects in this course. It is carefully commented and organized.
Contains:
- **imageAnalysis.py**: contains definition of MRI as an image. Used for plot the MRI and to compare two images.
Example of usage:

```
image_Young = ia.ImageProperties(dataset[0]**1.)
image_Old = ia.ImageProperties(dataset[1]**1.)

image_Young.compare(image_Old)
```

- **MachineLearningAlgorithms.py**: contains two classes - Features and Prediction.
In the Feature class you can find our implementation for feature extraction. We extract features for 2D and for 3D as well. We implement a `grid` and also a `threshold` operation. The `grid` operation extract the selected feature using the given grid as a mask. The `threshold`operation we ....

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

Meanwhile in the Prediction class you can find implementations of some linear regression algorithms as well as some implementations using libraries as sklearn. Besides that you can find the code for split the training data between training and validation dataset also our implementation of cross validation.
Linear Regression algorithms:
1. Basic linear regression (without regularization)
2. Lasso
3. Ridge regression
4. RidgeCV (cross validation)
5. SVR-Linear (Support Vector Regression with Linear kernel)


- **predict_final.py**: contains the main structure of our code, i. e., the whole workflow: load the training data, load the known predictions, divide the training data in training and validation, extract features, find the parameters for the selected model, load the test data, extract features, get predictions and write the result in a output file.

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

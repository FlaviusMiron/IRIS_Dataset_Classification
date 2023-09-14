# IRIS_Dataset_Classification
Classification of the IRIS dataset with a neural network and a soft-max logistic regression model made from scratch.
The IRIS dataset can be downloaded from here: https://archive.ics.uci.edu/dataset/53/iris and it consists of 150 examples of 3 subspecies of the Iris flower. The goel is to predict what subspecie a new flower is based on:
sepal length, sepal width, petal length and petal width. 

The subspecies are: 
Iris SetosaIris, Versicolour and Iris Virginica (more details on the link).

The "train_model.py" is responsible with training the neural network while with the "data_loader.py" I indended to make the data set more suitable for the neural net (see the docstring).
Achieves 100% accuracy on unseen data.

Required libraries are: numpy and matplotlib

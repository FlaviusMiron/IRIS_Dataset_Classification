"""
Data loader used to load the IRIS database, from the file "iris.data". It automatically re-fromats the data
and it can also rescale it for better usage in the network.
"""

import numpy as np


class InvalidRangeException(BaseException):
    pass

def format_data():
    """
    Change the format of the data into a more convenient form, meaning every example will form a list
    that contains 150 tuples of the form (features,label). "features" will be a (4,1) dimensional numpy array
    and "label" a (3,1) containg the class, in the form of [1,0,0] for Iris-setosa, [0,1,0] for Iris-versicolor
    and [0,0,1] for Iris-virginica. In this way it is more convenient for training and testing.
    """
    file = open("iris.data","r")
    data = file.readlines()
    reformatted_data = []

    for line in data:
        splitted_line = line.split(",")

        features = np.array([float(splitted_line[0]),float(splitted_line[1]),float(splitted_line[2]),float(splitted_line[3])])

        if splitted_line[4] == "Iris-setosa\n":
            label = np.array([1,0,0])
        elif splitted_line[4] == "Iris-versicolor\n":
            label = np.array([0,1,0])
        else:
            label = np.array([0,0,1])

        reformatted_data.append((features.reshape(-1,1),label.reshape(-1,1)))

    np.random.shuffle(reformatted_data)
    return reformatted_data

def rescale_formatted_data():
    """
    Uses min-max feature rescaling method in order to rescale the features between 0 and 1
    You can also opt to not rescale by modifiyng the "rescale" parameter of the "get_data" function, for experimentation.
    """
    rescale_formatted_data = []
    data = format_data()

    feature_values = []
    for features,label in data:
        for feature in features:
            feature_values.append(feature)

    highest_feature = max(feature_values)
    lowest_feature = min(feature_values)

    for feature,label in data:
        rescale_formatted_data.append(((feature-lowest_feature)/(highest_feature - lowest_feature),label))

    return rescale_formatted_data

def get_data(division_ratio : float = 0.8, rescale : bool = True):
    """Returns formatted and rescaled training and testing data. Parameter division_ratio has to be between 0.1 and 0.9."""
    
    # Checking the input
    if not isinstance(division_ratio,float):
        raise TypeError("Invalid type: division_ratio must be a float between 0.1 and 0.9.")
    
    if not isinstance(rescale,bool):
        raise TypeError("Invalid type: rescale has to be a boolean value.")
    
    if division_ratio > 0.9 or division_ratio < 0.1:
        raise InvalidRangeException("Invalid range: division_ratio must be between 0.1 and 0.9.")
    
    # Divide intro training and testing data, then return 
    if rescale:
        data = rescale_formatted_data()
    else:
        data = format_data()

    limit = int(division_ratio * len(data))
    training_data = data[:limit]
    test_data = data[limit:]

    return training_data,test_data


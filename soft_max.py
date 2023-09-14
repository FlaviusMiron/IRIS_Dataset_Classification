"""
A soft-max logistic regression model, implemented to solve the multi-class classification problem of the IRIS data set.
Minimizes the negative log-likelihood, cross entropy loss function by fitting a probability distribution to every example, and comparing it to the
ideal distribution. The highest logit is picked as the final answer. Uses batch gradient descent.
Prints the progress from 10 to 10 epochs (can be modified in line 52).
"""


import numpy as np
import matplotlib.pyplot as plt

class soft_max_logistic_regression():
    def __init__(self, classes, features):
        """Initialize various parameters."""
        self.num_classes = classes
        self.num_features = features

        self.theta_matrix = np.random.randn(classes, features) # Store parameters associated with each class

    def feed_forward(self, example):
        """Returns the predicted probability distribution for an example, as a numpy array"""
        output = np.zeros((self.num_classes,1))

        for index, thetas in zip(range(self.num_classes),self.theta_matrix):
            output[index] = np.exp(np.dot(thetas ,example))/sum([np.exp(np.dot(t,example)) for t in self.theta_matrix])

        return output

    def SGD(self,training_data, test_data = None, epochs = 100 ,learning_rate = 5):
        """Train the model by minimizing the negative log-likelihood cross entropy cost function, using stochastic gradient descent."""
        len_training_data = len(training_data)
        training_costs = []
        test_costs = []

        if test_data:
            len_test_data = len(test_data)

        
        for epoch in range(epochs):
            theta_matrix_gradients = np.zeros((self.num_classes, self.num_features))

            for index, thetas in zip(range(self.num_classes),self.theta_matrix):
                for example,label in training_data:
                    theta_matrix_gradients[index] = theta_matrix_gradients[index].reshape(1,4) + ((np.exp(np.dot(thetas ,example))/sum([np.exp(np.dot(t,example)) for t in self.theta_matrix])
                                                      - self.indicator(index,label)) * example).transpose()
                

            self.theta_matrix = self.theta_matrix - learning_rate * theta_matrix_gradients/len_training_data

                

            if epoch % 10 == 0:
                # Compute costs
                cost = sum([-np.log(np.exp(np.dot(self.theta_matrix[np.argmax(label)] ,features))/sum([np.exp(np.dot(t,features)) for t in self.theta_matrix]))
                             for features,label in training_data])/len_training_data
                training_costs.append(cost)
                # Evaluating
                
                if test_data:
                    guesses = self.evaluate(test_data)
                    print("Epoch {}: Guessed {} out of {} (soft-max)".format(epoch,guesses, len_test_data))
                    cost = sum([-np.log(np.exp(np.dot(self.theta_matrix[np.argmax(label)] ,features))/sum([np.exp(np.dot(t,features)) for t in self.theta_matrix]))
                                 for features,label in test_data])/len_test_data
                    test_costs.append(cost)

        plt.figure()
        plt.plot(training_costs)
        plt.title("Training cost (soft-max)")

        plt.figure()
        plt.plot(test_costs)
        plt.title("Testing cost (soft-max)")

    def evaluate(self, data):
        guesses = [(np.argmax(target),np.argmax(self.feed_forward(features)))  for features, target in data]
        return sum([x == y for (x,y) in guesses])

    def indicator(self, index, label):
        """The indicator function appears in the gradient of the cost functions with respect to each parameter group."""
        if label[index] == 1:
            return 1
        else:
            return 0


if __name__ == "__main__":
    import data_loader
    training_data, test_data = data_loader.get_data()

    model = soft_max_logistic_regression(3,4)
    model.SGD(training_data, test_data)
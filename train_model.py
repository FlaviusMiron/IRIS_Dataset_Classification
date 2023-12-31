"""
Script used to train the neural network and the softmax model. Showcases multiple model variants that can achieve 100%
accuracy on the unseen test_data (given some luck on the starting parameters :) )
"""
import data_loader
import neural_network
import soft_max

training_data, test_data = data_loader.get_data(0.8)

# model_nn = neural_network.MLP([4,12,3])
#model_nn.SGD(training_data, test_data, mini_batch_size = 20, epochs = 1000, learning_rate = 0.05)

model_nn = neural_network.MLP([4,30,3])
model_nn.SGD(training_data, test_data, mini_batch_size = 5, epochs = 100, learning_rate = 0.1)

model_sm = soft_max.soft_max_logistic_regression(3,4)
model_sm.SGD(training_data, test_data)

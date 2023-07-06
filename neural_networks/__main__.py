"""
Module runs and documents neural_network code.

@author Alexis Tudor
"""
import datetime
import warnings

from neural_network import NeuralNetwork

warnings.filterwarnings("ignore")

def nn_go(activation="s", learning_rate=0.1, iterations=100, threshold=1,
          test_split=.2, lambda_val=1):
    """
    Runs a neural network
    :param activation: type of activation function
    :param learning_rate: learning rate of neural network
    :param iterations: epochs to go
    :param threshold: When to stop
    :param test_split: Test/Training data split percentage
    :param lambda_val: Not implemented (but could be a weight penalty)
    """
    val = 0
    bad_result = 0.3808630393996248
    cont = True
    test_mse = 0
    test_accuracy = 0
    train_mse = 0
    train_accuracy = 0
    added = 0
    neural_network = NeuralNetwork("https://personal.utdallas.edu/" +
                                   "~art150530/occupancy.csv",
                                   activation=activation,
                                   learningrate=learning_rate,
                                   iterations=iterations,
                                   threshold=threshold,
                                   test_split=test_split,
                                   lambda_val=lambda_val)
    neural_network.train()
    test_mse, test_accuracy = neural_network.test()
    train_mse, train_accuracy = neural_network.test_on_training_data()
    if activation == "s":
        activation_string = "Sigmoid"
    elif activation == "t":
        activation_string = "Tanh"
    elif activation == "r":
        activation_string = "ReLu"
    else:
        activation_string = "N/A"
    file_local = open("results.csv", "a")
    file_local.write(str(datetime.datetime.now()) + "," +
               activation_string + "," +
               str(learning_rate) + "," +
               str(iterations) + "," +
               str(round(train_mse, 3)) + "," +
               str(round(train_accuracy*100)) + "," +
               str(round(test_mse, 3)) + "," +
               str(round(test_accuracy*100)) + "\n")
    file_local.close()

# 0.3808630393996248
if __name__ == '__main__':
    file = open("results.csv", "w")
    file.write("Time,Activation,Learning Rate,Iterations," +
               "Training Mean Squared Error (MSE),Training Accuracy," +
               "Testing MSE,Testing Accuracy\n")
    file.close()
    print(str(datetime.datetime.now()))
    # Activations
    print("Activations")
    print("Sigmoid")
    nn_go(activation="s")
    print("ReLu")
    nn_go(activation="r")
    print("Tanh")
    nn_go(activation="t")
    # Learning Rate
    print("Learning Rate")
    print(".2")
    nn_go(learning_rate=.2)
    print(".3")
    nn_go(learning_rate=.3)
    print(".4")
    nn_go(learning_rate=.4)
    print(".5")
    nn_go(learning_rate=.5)
    print("1")
    nn_go(learning_rate=1)
    # Iterations
    print("Iterations")
    print("10")
    nn_go(iterations=10)
    print("100")
    nn_go(iterations=100)
    print("1000")
    nn_go(iterations=1000)
    print(str(datetime.datetime.now()))

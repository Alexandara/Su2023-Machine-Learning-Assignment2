import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from random import seed
from random import random

class NeuralNetwork:
    """
    NeuralNetwork is a class that makes a neural network for the Occupancy
    dataset: https://archive.ics.uci.edu/dataset/357/occupancy+detection
    Data hosted: https://personal.utdallas.edu/~art150530/occupancy.csv
    """
    def __init__(self, location, activation="s"):
        self.load_data(location)
        self.preprocess_data()
        self.activation = activation
        seed(1)

    def load_data(self, location):
        """
        This method loads the data from a link into a pandas dataframe
        """
        self._data = pd.read_csv(location, skiprows=1, index_col=False,
                         names=["Number", "Date", "Temperature", "Humidity",
                                "Light", "CO2", "HumidityRatio", "Occupancy"],
                         usecols=["Temperature", "Humidity", "Light", "CO2",
                                  "HumidityRatio", "Occupancy"])

    def preprocess_data(self):
        """
        This method pre-processes a pandas dataframe in order to remove
        null lines and duplicates.
        """
        self._data.dropna(axis=0, how='any')
        self._data.drop_duplicates()
        self.X = self._data.iloc[:, :-1]
        self.y = self._data.iloc[:, -1]

    def train_test_split(self, test_percent=.2):
        """
        This method splits the data into train and test data
        :param test_percent: the percentage of the data to test on
        """
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=test_percent,
                             random_state=5)

    def initialize(self):
        """
        Creates a starting neural network
        """
        self.layer1 = \
            [[[random(),random(),random(),random(),random()], random()],
             [[random(),random(),random(),random(),random()], random()],
             [[random(),random(),random(),random(),random()], random()],
             [[random(),random(),random(),random(),random()], random()],
             [[random(),random(),random(),random(),random()], random()]]
        self.output = [[random(),random(),random(),random(),random()], random()]

    def train(self, iterations=100, threshold=1):
        """
        Create and train a neural network
        """
        loss = 0
        loss_count = 0
        for i in range(iterations):
            for index, row in self._data.iterrows():
                loss = loss + self.forward_propagation(row)
                loss_count = loss_count + 1
            loss = loss / loss_count
            if loss < threshold:
                break
            else:
                self.backpropagate(loss)
        return loss

    def test(self):
        """
        Test the neural network
        """

    def forward_propagation(self, data):
        """
        Perform forward propogation on one row.
        :param data: one row of the data
        :return: loss as a result of the final output
        """
        # Useful resource: https://towardsdatascience.com/neural-networks-backpropagation-by-dr-lihi-gur-arie-27be67d8fdce#:~:text=To%20update%20the%20weights%2C%20the,Learning%20rate%2C%20J%20%3D%20Cost.
        node_values = []
        for n in self.layer1:
            value = np.dot(n[0], data[:-1]) + n[1]
            node_values.append(value)
        node_values = self.activate(node_values, self.activation)
        final_value = np.dot(node_values, self.output[0]) + self.output[1]
        loss = ((data[-1:]["Occupancy"] - final_value) ** 2)/2
        return loss

    def backpropagate(self, loss):
        """
        Update the weights of the neural network.
        """


    def activate(self, values, activation):
        newvalues = []
        for value in values:
            if activation == "s":
                value = 1 / (1 + np.exp(-1 *value))
            elif activation == "t":
                value = np.tanh(value)
            elif activation == "r":
                if value <= 0:
                    value = 0
            newvalues.append(value)
        return newvalues

# For the record, it would be better to put this in a __main__.py
# file, or better yet, just test it with pytest, but I do not feel
# like doing that.
if __name__ == '__main__':
    neural_network = NeuralNetwork("https://personal.utdallas.edu/~art150530/occupancy.csv", activation="r")
    neural_network.train_test_split()
    neural_network.initialize()
    neural_network.train()
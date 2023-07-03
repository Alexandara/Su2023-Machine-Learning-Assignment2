"""
Module creates a neural network for the Summer 2023 Machine Learning course at
the University of Texas at Dallas
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class NeuralNetwork:
    """
    NeuralNetwork is a class that makes a neural network for the Occupancy
    dataset: https://archive.ics.uci.edu/dataset/357/occupancy+detection
    Data hosted: https://personal.utdallas.edu/~art150530/occupancy.csv
    """
    def __init__(self, location, activation="s", learningrate=0.1,
                 iterations=100, threshold=1, test_split=.2,
                 lambda_val=1):
        self.activation = activation
        self.learning_rate = learningrate
        self.iterations = iterations
        self.threshold = threshold
        self.test_percent = test_split
        self.lambda_val = lambda_val
        self.load_data(location)
        self.preprocess_data()
        self.train_test_split()
        self.layer1 = np.random.randn(5, 5)  # 5x5 matrix
        self.outputlayer = np.random.randn(5, 1)
        self.node_values = []
        self.unactivated_hidden = []
        self.unactivated_output = []
        self.output = [0]
        self.d_hidden = []
        self.d_output = []
        np.random.seed(1)

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
        dataframe = self._data.iloc[:, :-1]
        self.x_data = (dataframe - dataframe.min()) / \
                      (dataframe.max() - dataframe.min())
        self.y_data = self._data.iloc[:, -1]

    def train_test_split(self):
        """
        This method splits the data into train and test data
        :param test_percent: the percentage of the data to test on
        """
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.x_data, self.y_data, test_size=self.test_percent,
                             random_state=5)
        self.training = pd.concat([self.x_train, self.y_train], axis=1, join='inner')
        self.testing = pd.concat([self.x_test, self.y_test], axis=1, join='inner')

    def train(self):
        """
        Train a neural network
        """
        for _ in range(self.iterations):
            for _, row in self.training.iterrows():
                prediction = self.forward_propagation(row)
                value = prediction - row['Occupancy']
                self.backpropagate(row, value)
                self.update()

    def test(self):
        """
        Test the neural network
        """
        predictions = []
        for _, row in self.testing.iterrows():
            pred = self.predict(row)
            predictions.append(pred)
        return self.loss(predictions, self.testing["Occupancy"].to_list())

    def forward_propagation(self, data):
        """
        Perform forward propogation on one row.
        :param data: one row of the data
        :return: the final prediction
        """
        # Useful resource: https://towardsdatascience.com/neural-networks
        # -backpropagation-by-dr-lihi-gur-arie-27be67d8fdce#:~:text=To%20update
        # %20the%20weights%2C%20the,Learning%20rate%2C%20J%20%3D%20Cost.
        self.unactivated_hidden = np.dot(self.layer1.T, data[:-1].T)
        self.node_values = self.activate(self.unactivated_hidden, self.activation)
        self.unactivated_output = np.dot(self.outputlayer.T, self.node_values)
        self.output = self.activate(self.unactivated_output, self.activation)
        return self.output

    def backpropagate(self, data, prediction):
        """
        Update the weights of the neural network.
        :param data: one row of the data
        """
        # Useful Resource: https://medium.com/@qempsil0914/implement-neural-
        # network-without-using-deep-learning-libraries-step-by-step-tutorial-
        # python3-e2aa4e5766d1
        x_data = data[:-1]
        y_data = data[-1:]
        num = x_data.shape[0]
        deltaoutput = prediction - y_data["Occupancy"]
        dunactivated_output = np.multiply(deltaoutput,
                                          self.activate(self.unactivated_output,
                                                        self.activation,
                                                        derivative=True))
        self.d_hidden = (1 / num) * np.sum(np.multiply(self.node_values,
                                                     dunactivated_output))
        deltahidden = deltaoutput * self.outputlayer * \
                 self.activate(self.unactivated_hidden, self.activation, derivative=True)
        self.d_output = np.array((1 / num) * np.dot(x_data.T, deltahidden.T))

    def activate(self, values, activation, derivative=False):
        """
        A function that runs an activation function on a set of values.
        :param values: array of values to be activated.
        :param activation:
        :return:
        """
        newvalues = []
        if activation == "s":
            newvalues = 1 / (1 + np.exp(-values))
            if derivative:
                newvalues = newvalues * (1 - newvalues)
        elif activation == "t":
            newvalues = np.tanh(values)
            if derivative:
                newvalues = 1. - newvalues ** 2
        elif activation == "r":
            for value in values:
                if not derivative:
                    if value <= 0:
                        newvalues.append(0)
                    else:
                        newvalues.append(value)
                elif derivative:
                    if value < 0:
                        newvalues.append(0)
                    else:
                        newvalues.append(1)

        return newvalues

    def loss(self, predictions, y_data):
        """
        Function that calculates the loss of a set of predictions per the truth
        in the data.
        :param predictions: a set of predicted values of y
        :param y: the actual values of y
        :return:
        """
        num = len(y_data)
        loss = 0
        for i in range(num):
            loss = loss + ((predictions[i] - y_data[i]) ** 2)
        return loss / num

    def update(self):
        """
        Function that updates the weights of the neural network after
        backpropagation.
        """
        for i in range(4):
            for j in range(4):
                self.layer1[i][j] = self.layer1[i][j] - self.learning_rate * self.d_hidden
        for i in range(len(self.d_output)):
            self.outputlayer[i][0] = self.outputlayer[i][0] - self.learning_rate * self.d_output[i]

    def predict(self, data):
        """
        For a single instance row of data, runs the model to predict it.
        :param data: a row of the data
        :return: prediction, 1 or 0
        """
        prediction = self.forward_propagation(data)
        if prediction[0] > .5:
            return 1
        return 0


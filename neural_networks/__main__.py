from neural_network import NeuralNetwork

def nn_go(activation="s", learningrate=0.1, iterations=100, threshold=1,
          test_split=.2, lambda_val=1):
    neural_network = NeuralNetwork("https://personal.utdallas.edu/" +
                                   "~art150530/occupancy.csv",
                                   activation=activation,
                                   learningrate=learningrate,
                                   iterations=iterations,
                                   threshold=threshold,
                                   test_split=test_split,
                                   lambda_val=lambda_val)
    neural_network.train()
    return neural_network.test()

# 0.3808630393996248
if __name__ == '__main__':
    val = 0
    result = 0.3808630393996248
    while result == 0.3808630393996248 and val < 10:
        result = nn_go()
    print(result)
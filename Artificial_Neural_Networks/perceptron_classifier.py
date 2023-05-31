import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl
import os


from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

DATA_PERCEPTRON_PATH = os.getenv("DATA_PERCEPTRON_PATH")
DATA_PERCEPTRON_FILE = os.getenv("DATA_PERCEPTRON_FILE")


class Perceptron:
    def __init__(self):
        self.data_perceptron_file = np.loadtxt(
            f"{DATA_PERCEPTRON_PATH}{DATA_PERCEPTRON_FILE}"
        )
        self.datapoints = self.data_perceptron_file[:, :2]
        self.labels = self.data_perceptron_file[:, 2].reshape(
            self.data_perceptron_file.shape[0], 1
        )
        self.first_minimum_dimension = int(
            os.getenv("FIRST_MINIMUM_DIMENSION")
        )
        self.first_maximum_dimension = int(
            os.getenv("FIRST_MAXIMUM_DIMENSION")
        )
        self.second_minimum_dimension = int(
            os.getenv("SECOND_MINIMUM_DIMENSION")
        )
        self.second_maximum_dimension = int(
            os.getenv("SECOND_MAXIMUM_DIMENSION")
        )
        self.output_neurons_number = self.labels.shape[1]
        self.first_perceptron_dimension = [
            self.first_minimum_dimension, self.first_maximum_dimension
        ]
        self.second_perceptron_dimension = [
            self.second_minimum_dimension, self.second_maximum_dimension
        ]
        self.perceptron = nl.net.newp(
            [
                self.first_perceptron_dimension,
                self.second_perceptron_dimension
            ],
            self.output_neurons_number
        )
        self.epochs = int(os.getenv('EPOCHS'))
        self.show = int(os.getenv("SHOW"))
        self.lr = float(os.getenv('LR'))
        self.error_progess = self.perceptron.train(
            self.datapoints, self.labels,
            epochs=self.epochs, show=self.show,
            lr=self.lr
        )

    def plot_input_data(self):
        plt.figure()
        plt.scatter(
            self.datapoints[:, 0], self.datapoints[:, 1]
        )
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Input Data')

    def plot_output_data(self):
        plt.figure()
        plt.plot(self.error_progess)
        plt.xlabel('Epochs Number')
        plt.ylabel('Error Training')
        plt.title('Training Error Progress')
        plt.grid()


perceptron = Perceptron()
perceptron.plot_input_data()
perceptron.plot_output_data()
plt.show()

# https://medium.com/@thomascountz/19-line-line-by-line-python-perceptron-b6f113b161f3

"""
MIT License
Copyright (c) 2018 Thomas Countz
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        print(f"Init! Threshold: {threshold}, Learning rate: {learning_rate}, Number of inputs = {no_of_inputs}")
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        print(f"Prediction for: {inputs}")
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        print(f"Activation: {activation}")
        return activation

    def train(self, training_inputs, labels):
        print("Train")
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                print(f"- - -")
                print(f"Inputs: {inputs} --- Label: {label}")
                print(f"Weights before: {self.weights}")
                prediction = self.predict(inputs)
                weight_array_addition = self.learning_rate * (label - prediction) * inputs
                weight_single_addition = self.learning_rate * (label - prediction)
                print(f"Addition for inputs: {weight_array_addition}")
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                print(f"Addition for single add in: {weight_single_addition}")
                self.weights[0] += self.learning_rate * (label - prediction)
                print(f"Weights after: {self.weights}")

class PerceptronTest():

    def test_mimics_logical_and(self):
        weights = np.array([-1, 1, 1])

        a = 1
        b = 1
        inputs = np.array([a, b])

        perceptron = Perceptron(inputs.size)
        perceptron.weights = weights

        output = perceptron.predict(inputs)
        print(output, a & b)

    def test_trains_for_logical_and(self):
        labels = np.array([1, 0, 0, 0])
        input_matrix = []
        input_matrix.append(np.array([1, 1]))
        input_matrix.append(np.array([1, 0]))
        input_matrix.append(np.array([0, 1]))
        input_matrix.append(np.array([0, 0]))

        perceptron = Perceptron(2, threshold=10, learning_rate=1)
        perceptron.train(input_matrix, labels)

        a = 1
        b = 1
        inputs = np.array([a, b])

        output = perceptron.predict(inputs)
        print(output, a & b)

if __name__ == '__main__':
        unittest = PerceptronTest()
        unittest.test_mimics_logical_and()
        unittest.test_trains_for_logical_and()

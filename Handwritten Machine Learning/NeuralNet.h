#pragma once
#include "JMatrix.h"

class NeuralNet {
private:
	// includes input and output layers
	int layers = 5;
	int neuronsPerLayer[5] = { 784, 300, 200, 100, 10 };

	JMatrix<float>* weightMatrices;
	JMatrix<float>* biasVectors;

	// distance from 0
	float initValueRange = 8;

public:
	NeuralNet() {
		weightMatrices = new JMatrix<float>[layers - 1];
		biasVectors = new JMatrix<float>[layers - 1];

		for (int i = 0; i < layers - 1; i++) {
			int inputVectorLength = neuronsPerLayer[i];
			int outputVectorLength = neuronsPerLayer[i + 1];

			// matrix initialisation
			weightMatrices[i] = JMatrix<float>(inputVectorLength, outputVectorLength);
			for (int row = 0; row < outputVectorLength; row++) {
				for (int col = 0; col < inputVectorLength; col++) {
					float randValue = ((rand() / (float)RAND_MAX) * initValueRange * 2) - initValueRange;

					weightMatrices[i].setValue(col, row, randValue);
				}
			}

			// bias vector initialisation
			biasVectors[i] = JMatrix<float>(1, outputVectorLength);
			for (int row = 0; row < outputVectorLength; row++) {
				float randValue = ((rand() / (float)RAND_MAX) * initValueRange * 2) - initValueRange;

				biasVectors[i].setValue(0, row, randValue);
			}
		}
	}

	~NeuralNet() {
		delete[] weightMatrices;
		delete[] biasVectors;
	}

	NeuralNet(const NeuralNet& copy) {
		layers = copy.layers;
		// Add later after implementing dynamic neural net structure
		//neuronsPerLayer = copy.neuronsPerLayer

		for (int i = 0; i < layers - 1; i++) {
			weightMatrices[i] = copy.weightMatrices[i];
			biasVectors[i] = copy.biasVectors[i];
		}
		
		initValueRange = copy.initValueRange;
	}

	const NeuralNet& operator=(const NeuralNet& copy) {
		layers = copy.layers;
		// Add later after implementing dynamic neural net structure
		//neuronsPerLayer = copy.neuronsPerLayer

		for (int i = 0; i < layers - 1; i++) {
			weightMatrices[i] = copy.weightMatrices[i];
			biasVectors[i] = copy.biasVectors[i];
		}

		initValueRange = copy.initValueRange;
	}

	float sigmoidFunction(float value) {
		float e = 2.71828;

		return 1 / (1 + pow(e, -value));
	}

	JMatrix<float> computeOutput(const JMatrix<float>& inputVector) {
		JMatrix<float> vector = inputVector;

		for (int i = 0; i < layers - 1; i++) {
			vector = weightMatrices[i].multiply(vector);
			vector = biasVectors[i].add(vector);

			for (int n = 0; n < neuronsPerLayer[i + 1]; n++) {
				float sigmoidValue = sigmoidFunction(vector.getValue(0, n));
				vector.setValue(0, n, sigmoidValue);
			}
		}

		return vector;
	}
};
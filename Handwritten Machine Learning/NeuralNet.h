#pragma once
#include "JMatrix.h"

class NeuralNet {
private:
	// includes input and output layers
	int layers = 5;
	int neuronsPerLayer[5] = { 784, 300, 200, 100, 10 };

	JMatrix<float>* weightMatrices;
	JMatrix<float>* biasVectors;

	JMatrix<float>* neuronLayers;

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

		neuronLayers = new JMatrix<float>[layers];
		for (int i = 0; i < layers; i++) {
			neuronLayers[i] = JMatrix<float>(1, neuronsPerLayer[i]);
		}
	}

	~NeuralNet() {
		delete[] weightMatrices;
		delete[] biasVectors;

		delete[] neuronLayers;
	}

	NeuralNet(const NeuralNet& copy) {
		layers = copy.layers;
		// Add later after implementing dynamic neural net structure
		//neuronsPerLayer = copy.neuronsPerLayer

		weightMatrices = new JMatrix<float>[layers - 1];
		biasVectors = new JMatrix<float>[layers - 1];
		for (int i = 0; i < layers - 1; i++) {
			weightMatrices[i] = copy.weightMatrices[i];
			biasVectors[i] = copy.biasVectors[i];
		}

		neuronLayers = new JMatrix<float>[layers];
		for (int i = 0; i < layers; i++) {
			neuronLayers[i] = copy.neuronLayers[i];
		}
		
		initValueRange = copy.initValueRange;
	}

	const NeuralNet& operator=(const NeuralNet& copy) {
		delete[] weightMatrices;
		delete[] biasVectors;

		delete[] neuronLayers;

		layers = copy.layers;
		// Add later after implementing dynamic neural net structure
		//neuronsPerLayer = copy.neuronsPerLayer

		weightMatrices = new JMatrix<float>[layers - 1];
		biasVectors = new JMatrix<float>[layers - 1];

		for (int i = 0; i < layers - 1; i++) {
			weightMatrices[i] = copy.weightMatrices[i];
			biasVectors[i] = copy.biasVectors[i];
		}

		neuronLayers = new JMatrix<float>[layers];
		for (int i = 0; i < layers; i++) {
			neuronLayers[i] = copy.neuronLayers[i];
		}

		initValueRange = copy.initValueRange;
	}

	float sigmoidFunction(float value) {
		float e = 2.71828;

		return 1 / (1 + pow(e, -value));
	}

	int getLayerCount() {
		return layers;
	}

	JMatrix<float>& getWeightMatrix(int index) {
		if (index < 0 || index >= layers - 1) {
			throw "out of bounds";
		}

		return weightMatrices[index];
	}

	JMatrix<float>& getBiasVector(int index) {
		if (index < 0 || index >= layers - 1) {
			throw "out of bounds";
		}

		return biasVectors[index];
	}

	JMatrix<float>& getNeuronLayer(int index) {
		if (index < 0 || index >= layers) {
			throw "out of bounds";
		}

		return neuronLayers[index];
	}

	JMatrix<float>& getInputLayer() {
		return neuronLayers[0];
	}

	JMatrix<float>& getOutputLayer() {
		return neuronLayers[layers - 1];
	}

	void run() {
		for (int i = 0; i < layers - 1; i++) {
			JMatrix<float> vector = neuronLayers[i];

			vector = weightMatrices[i].multiply(vector);
			vector = biasVectors[i].add(vector);

			for (int n = 0; n < neuronsPerLayer[i + 1]; n++) {
				float sigmoidValue = sigmoidFunction(vector.getValue(0, n));
				vector.setValue(0, n, sigmoidValue);
			}

			neuronLayers[i + 1] = vector;
		}
	}
};
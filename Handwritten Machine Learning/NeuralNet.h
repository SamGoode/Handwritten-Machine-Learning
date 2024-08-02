#pragma once
#include "JMatrix.h"

class NeuralNet {
private:
	// includes input and output layers
	int layers = 5;
	int neuronsPerLayer[5] = { 784, 300, 200, 100, 10 };

	JMatrix<float>* weightMatrices;
	JVector<float>* biasVectors;

	JVector<float>* preActivationSums;
	JVector<float>* neuronLayers;

	// distance from 0
	float initValueRange = 16;

public:
	NeuralNet() {
		weightMatrices = new JMatrix<float>[layers - 1];
		biasVectors = new JVector<float>[layers - 1];
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
			biasVectors[i] = JVector<float>(outputVectorLength);
			for (int index = 0; index < biasVectors[i].getSize(); index++) {
				float randValue = ((rand() / (float)RAND_MAX) * initValueRange * 2) - initValueRange;

				biasVectors[i].setValue(index, randValue);
			}
		}

		preActivationSums = new JVector<float>[layers - 1];
		for (int i = 0; i < layers - 1; i++) {
			preActivationSums[i] = JVector<float>(neuronsPerLayer[i + 1]);
		}

		neuronLayers = new JVector<float>[layers];
		for (int i = 0; i < layers; i++) {
			neuronLayers[i] = JVector<float>(neuronsPerLayer[i]);
		}
	}

	~NeuralNet() {
		delete[] weightMatrices;
		delete[] biasVectors;

		delete[] preActivationSums;
		delete[] neuronLayers;
	}

	NeuralNet(const NeuralNet& copy) {
		layers = copy.layers;
		// Add later after implementing dynamic neural net structure
		//neuronsPerLayer = copy.neuronsPerLayer

		weightMatrices = new JMatrix<float>[layers - 1];
		biasVectors = new JVector<float>[layers - 1];
		for (int i = 0; i < layers - 1; i++) {
			weightMatrices[i] = copy.weightMatrices[i];
			biasVectors[i] = copy.biasVectors[i];
		}

		preActivationSums = new JVector<float>[layers - 1];
		for (int i = 0; i < layers - 1; i++) {
			preActivationSums[i] = copy.preActivationSums[i];
		}

		neuronLayers = new JVector<float>[layers];
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
		biasVectors = new JVector<float>[layers - 1];

		for (int i = 0; i < layers - 1; i++) {
			weightMatrices[i] = copy.weightMatrices[i];
			biasVectors[i] = copy.biasVectors[i];
		}

		preActivationSums = new JVector<float>[layers - 1];
		for (int i = 0; i < layers - 1; i++) {
			preActivationSums[i] = copy.preActivationSums[i];
		}

		neuronLayers = new JVector<float>[layers];
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

	JVector<float>& getBiasVector(int index) {
		if (index < 0 || index >= layers - 1) {
			throw "out of bounds";
		}

		return biasVectors[index];
	}

	JVector<float>& getPreActivation(int index) {
		if (index < 0 || index >= layers - 1) {
			throw "out of bounds";
		}

		return preActivationSums[index];
	}

	JVector<float>& getNeuronLayer(int index) {
		if (index < 0 || index >= layers) {
			throw "out of bounds";
		}

		return neuronLayers[index];
	}

	JVector<float>& getInputLayer() {
		return neuronLayers[0];
	}

	JVector<float>& getOutputLayer() {
		return neuronLayers[layers - 1];
	}

	void run() {
		for (int i = 0; i < layers - 1; i++) {
			JVector<float> vector = neuronLayers[i];

			vector = weightMatrices[i].multiply(vector);
			vector = biasVectors[i].add(vector);

			preActivationSums[i] = vector;

			for (int n = 0; n < neuronsPerLayer[i + 1]; n++) {
				float sigmoidValue = sigmoidFunction(vector[n]);
				vector.setValue(n, sigmoidValue);
			}

			neuronLayers[i + 1] = vector;
		}
	}
};
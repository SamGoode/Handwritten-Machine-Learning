#pragma once
#include "JMatrix.h"
#include "JVector.h"

class NeuralNet {
private:
	float initValueRange;
	float initValueRangeBias;

	JVector<float> inputLayer;
	JVector<float> outputLayer;
	int hiddenLayerCount;
	JVector<float>* hiddenLayers;

	JMatrix<float>* weightMatrices;
	JVector<float>* biasVectors;

	JVector<float>* layerGradients;
	JMatrix<float>* weightGradients;
	JVector<float>* biasGradients;

public:
	NeuralNet() {}
	NeuralNet(int layerCount, ...);
	~NeuralNet();

	NeuralNet(const NeuralNet& copy);

	const NeuralNet& operator=(const NeuralNet& copy);

	static float sigmoidFunction(float value) { return 1 / (1 + pow(2.71828f, -value)); }
	static float derivativeSigmoid(float input) { return 1 / (2 + pow(2.71828f, input) + pow(2.71828f, -input)); }
	static float derivativePostSigmoid(float input) { return input * (1 - input); }
	static float derivativeCost(float givenValue, float desiredValue) { return (givenValue - desiredValue); }

	static void applySigmoid(JVector<float>& vector) {
		for (int i = 0; i < vector.getSize(); i++) {
			vector[i] = sigmoidFunction(vector[i]);
		}
	}

	int getLayerCount() { return hiddenLayerCount + 2; }
	JVector<float>& getInputLayer() { return inputLayer; }
	JVector<float>& getOutputLayer() { return outputLayer; }
	JVector<float>& getHiddenLayer(int index) { if (index < 0 || index >= hiddenLayerCount) { throw "out of bounds"; } return hiddenLayers[index]; }

	JMatrix<float>& getWeightMatrix(int index) { if (index < 0 || index >= hiddenLayerCount + 1) { throw "out of bounds"; } return weightMatrices[index]; }
	JVector<float>& getBiasVector(int index) { if (index < 0 || index >= hiddenLayerCount + 1) { throw "out of bounds"; } return biasVectors[index]; }

	void run();
	void train(int expectedValue);
	void applyGradients(float learningRate, int batchSize);
};
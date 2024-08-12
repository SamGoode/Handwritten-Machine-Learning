#pragma once
#include "JMatrix.h"
#include "JVector.h"
#include "PixelGrid.h"

class NeuralNet {
private:
	float initWeightRange;
	float initBiasRange;

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
	NeuralNet(int layerCount, int* neuronsPerLayer);
	~NeuralNet();

	NeuralNet(const NeuralNet& copy);
	const NeuralNet& operator=(const NeuralNet& copy);

	int getLayerCount() { return hiddenLayerCount + 2; }
	JVector<float>& getInputLayer() { return inputLayer; }
	JVector<float>& getOutputLayer() { return outputLayer; }
	JVector<float>& getHiddenLayer(int index) { if (index < 0 || index >= hiddenLayerCount) { throw "out of bounds"; } return hiddenLayers[index]; }

	JMatrix<float>& getWeightMatrix(int index) { if (index < 0 || index >= hiddenLayerCount + 1) { throw "out of bounds"; } return weightMatrices[index]; }
	JVector<float>& getBiasVector(int index) { if (index < 0 || index >= hiddenLayerCount + 1) { throw "out of bounds"; } return biasVectors[index]; }

	void loadInGrid(PixelGrid& pixelGrid);

	static void randomiseMatrix(JMatrix<float>& matrix, float randRange);
	static void randomiseVector(JVector<float>& vector, float randRange);

	void run();
	void train(int expectedValue);
	void applyGradients(float learningRate, int batchSize);
};
#include "NeuralNet.h"

NeuralNet::NeuralNet(int layerCount, ...) {
	int* neuronsPerLayer = new int[layerCount];
	for (int i = 0; i < layerCount; i++) {
		neuronsPerLayer[i] = *(&layerCount + 2 + i * 2);
	}

	if (layerCount < 2) {
		throw "Not enough args for input and output layers";
	}

	// distance from 0
	initValueRange = 1.f;
	initValueRangeBias = 1.f;

	inputLayer = JVector<float>(neuronsPerLayer[0]);
	outputLayer = JVector<float>(neuronsPerLayer[layerCount - 1]);

	hiddenLayerCount = layerCount - 2;
	hiddenLayers = new JVector<float>[hiddenLayerCount];
	for (int i = 0; i < hiddenLayerCount; i++) {
		hiddenLayers[i] = JVector<float>(neuronsPerLayer[i + 1]);
	}

	weightMatrices = new JMatrix<float>[hiddenLayerCount + 1];
	biasVectors = new JVector<float>[hiddenLayerCount + 1];
	for (int i = 0; i < hiddenLayerCount + 1; i++) {
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
			float randValue = ((rand() / (float)RAND_MAX) * initValueRangeBias * 2) - initValueRangeBias;

			biasVectors[i].setValue(index, randValue);
		}
	}

	layerGradients = new JVector<float>[hiddenLayerCount + 1];
	weightGradients = new JMatrix<float>[hiddenLayerCount + 1];
	biasGradients = new JVector<float>[hiddenLayerCount + 1];
	for (int i = 0; i < hiddenLayerCount + 1; i++) {
		int inputVectorLength = neuronsPerLayer[i];
		int outputVectorLength = neuronsPerLayer[i + 1];

		layerGradients[i] = JVector<float>(outputVectorLength);
		weightGradients[i] = JMatrix<float>(inputVectorLength, outputVectorLength);
		biasGradients[i] = JVector<float>(outputVectorLength);

		layerGradients[i].setAllValues(0);
		weightGradients[i].setAllValues(0);
		biasGradients[i].setAllValues(0);
	}

	delete[] neuronsPerLayer;
}

NeuralNet::NeuralNet(int layerCount, int* neuronsPerLayer) {
	if (layerCount < 2) {
		throw "Not enough args for input and output layers";
	}

	// distance from 0
	initValueRange = 1.f;
	initValueRangeBias = 1.f;

	inputLayer = JVector<float>(neuronsPerLayer[0]);
	outputLayer = JVector<float>(neuronsPerLayer[layerCount - 1]);

	hiddenLayerCount = layerCount - 2;
	hiddenLayers = new JVector<float>[hiddenLayerCount];
	for (int i = 0; i < hiddenLayerCount; i++) {
		hiddenLayers[i] = JVector<float>(neuronsPerLayer[i + 1]);
	}

	weightMatrices = new JMatrix<float>[hiddenLayerCount + 1];
	biasVectors = new JVector<float>[hiddenLayerCount + 1];
	for (int i = 0; i < hiddenLayerCount + 1; i++) {
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
			float randValue = ((rand() / (float)RAND_MAX) * initValueRangeBias * 2) - initValueRangeBias;

			biasVectors[i].setValue(index, randValue);
		}
	}

	layerGradients = new JVector<float>[hiddenLayerCount + 1];
	weightGradients = new JMatrix<float>[hiddenLayerCount + 1];
	biasGradients = new JVector<float>[hiddenLayerCount + 1];
	for (int i = 0; i < hiddenLayerCount + 1; i++) {
		int inputVectorLength = neuronsPerLayer[i];
		int outputVectorLength = neuronsPerLayer[i + 1];

		layerGradients[i] = JVector<float>(outputVectorLength);
		weightGradients[i] = JMatrix<float>(inputVectorLength, outputVectorLength);
		biasGradients[i] = JVector<float>(outputVectorLength);

		layerGradients[i].setAllValues(0);
		weightGradients[i].setAllValues(0);
		biasGradients[i].setAllValues(0);
	}
}

NeuralNet::~NeuralNet() {
	delete[] hiddenLayers;

	delete[] weightMatrices;
	delete[] biasVectors;

	delete[] layerGradients;
	delete[] weightGradients;
	delete[] biasGradients;
}

NeuralNet::NeuralNet(const NeuralNet& copy) {
	initValueRange = copy.initValueRange;
	initValueRangeBias = copy.initValueRangeBias;

	inputLayer = copy.inputLayer;
	outputLayer = copy.outputLayer;

	hiddenLayerCount = copy.hiddenLayerCount;
	hiddenLayers = new JVector<float>[hiddenLayerCount];
	for (int i = 0; i < hiddenLayerCount; i++) {
		hiddenLayers[i] = copy.hiddenLayers[i];
	}

	weightMatrices = new JMatrix<float>[hiddenLayerCount + 1];
	biasVectors = new JVector<float>[hiddenLayerCount + 1];
	for (int i = 0; i < hiddenLayerCount + 1; i++) {
		weightMatrices[i] = copy.weightMatrices[i];
		biasVectors[i] = copy.biasVectors[i];
	}

	layerGradients = new JVector<float>[hiddenLayerCount + 1];
	weightGradients = new JMatrix<float>[hiddenLayerCount + 1];
	biasGradients = new JVector<float>[hiddenLayerCount + 1];

	for (int i = 0; i < hiddenLayerCount + 1; i++) {
		layerGradients[i] = copy.layerGradients[i];
		weightGradients[i] = copy.weightGradients[i];
		biasGradients[i] = copy.biasGradients[i];
	}
}

const NeuralNet& NeuralNet::operator=(const NeuralNet& copy) {
	delete[] hiddenLayers;

	delete[] weightMatrices;
	delete[] biasVectors;

	delete[] layerGradients;
	delete[] weightGradients;
	delete[] biasGradients;

	initValueRange = copy.initValueRange;
	initValueRangeBias = copy.initValueRangeBias;

	inputLayer = copy.inputLayer;
	outputLayer = copy.outputLayer;

	hiddenLayerCount = copy.hiddenLayerCount;
	hiddenLayers = new JVector<float>[hiddenLayerCount];
	for (int i = 0; i < hiddenLayerCount; i++) {
		hiddenLayers[i] = copy.hiddenLayers[i];
	}

	weightMatrices = new JMatrix<float>[hiddenLayerCount + 1];
	biasVectors = new JVector<float>[hiddenLayerCount + 1];
	for (int i = 0; i < hiddenLayerCount + 1; i++) {
		weightMatrices[i] = copy.weightMatrices[i];
		biasVectors[i] = copy.biasVectors[i];
	}

	layerGradients = new JVector<float>[hiddenLayerCount + 1];
	weightGradients = new JMatrix<float>[hiddenLayerCount + 1];
	biasGradients = new JVector<float>[hiddenLayerCount + 1];
	for (int i = 0; i < hiddenLayerCount + 1; i++) {
		layerGradients[i] = copy.layerGradients[i];
		weightGradients[i] = copy.weightGradients[i];
		biasGradients[i] = copy.biasGradients[i];
	}

	return *this;
}

void NeuralNet::run() {
	// For special case where there are no hidden layers
	if (hiddenLayerCount == 0) {
		outputLayer.copy(weightMatrices[0].multiply(inputLayer)).addOn(biasVectors[0]);
		applySigmoid(outputLayer);

		return;
	}

	// Compute Input Layer into first Hidden layer
	hiddenLayers[0].copy(weightMatrices[0].multiply(inputLayer)).addOn(biasVectors[0]);
	applySigmoid(hiddenLayers[0]);

	// Compute Hidden Layers
	for (int i = 0; i < hiddenLayerCount - 1; i++) {
		hiddenLayers[i + 1].copy(weightMatrices[i + 1].multiply(hiddenLayers[i])).addOn(biasVectors[i + 1]);
		applySigmoid(hiddenLayers[i + 1]);
	}

	// Compute last Hidden Layer into Output Layer
	outputLayer.copy(weightMatrices[hiddenLayerCount].multiply(hiddenLayers[hiddenLayerCount - 1])).addOn(biasVectors[hiddenLayerCount]);
	//applySigmoid(outputLayer);
	applySoftMax(outputLayer);
}

void NeuralNet::train(int expectedValue) {
	// Calculate deltas at output layer
	layerGradients[hiddenLayerCount].copy(outputLayer);
	for (int i = 0; i < outputLayer.getSize(); i++) {
		// Index of output layer matches its corresponding value (0-9)
		float desiredConfidenceValue = 0;
		if (i == expectedValue) {
			desiredConfidenceValue = 1;
		}

		float outputNeuronValue = outputLayer[i];
		float value = outputLayer[i] - desiredConfidenceValue;
		//float derivativeCostValue = derivativeCost(outputNeuronValue, desiredConfidenceValue);

		//float derivativeSigmoidValue = derivativePostSigmoid(outputNeuronValue);
		//float value = derivativeSigmoidValue * derivativeCostValue;

		//float derivativeSoftMaxValue = derivativeSoftMax(i);
		//float value = derivativeSoftMaxValue * derivativeCostValue;

		layerGradients[hiddenLayerCount][i] = value;
	}

	// Calculating deltas at hidden layers
	for (int i = 0; i < hiddenLayerCount; i++) {
		int layerIndex = hiddenLayerCount - 1 - i;
		JMatrix<float>& weightMatrix = weightMatrices[layerIndex + 1];
		layerGradients[layerIndex].copy(weightMatrix.transposedMultiply(layerGradients[layerIndex + 1]));

		for (int index = 0; index < layerGradients[layerIndex].getSize(); index++) {
			float hiddenNeuronValue = hiddenLayers[layerIndex][index];
			float derivativeSigmoidValue = derivativePostSigmoid(hiddenNeuronValue);

			float value = layerGradients[layerIndex][index] * derivativeSigmoidValue;
			layerGradients[layerIndex][index] = value;
		}
	}

	// Computing gradient vectors for weights and biases

	// For special cases where there are no hidden layers
	if (hiddenLayerCount == 0) {
		// Compute partial derivatives between Input Layer and Output Layer
		for (int row = 0; row < weightMatrices[0].getRowCount(); row++) {
			float activationDerivative = layerGradients[0][row];

			for (int col = 0; col < weightMatrices[0].getColumnCount(); col++) {
				float weightDerivative = inputLayer[col] * activationDerivative;
				weightGradients[0].addValue(col, row, weightDerivative);
			}

			float biasDerivative = activationDerivative;
			biasGradients[0].addValue(row, biasDerivative);
		}

		return;
	}

	// Compute partial derivatives between Input Layer and first Hidden Layer
	for (int row = 0; row < weightMatrices[0].getRowCount(); row++) {
		float activationDerivative = layerGradients[0][row];

		for (int col = 0; col < weightMatrices[0].getColumnCount(); col++) {
			float weightDerivative = inputLayer[col] * activationDerivative;
			weightGradients[0].addValue(col, row, weightDerivative);
		}

		float biasDerivative = activationDerivative;
		biasGradients[0].addValue(row, biasDerivative);
	}

	// Compute partial derivatives between hidden layers
	for (int i = 0; i < hiddenLayerCount - 1; i++) {
		JMatrix<float>& weightMatrix = weightMatrices[i + 1];
		JVector<float>& biasVector = biasVectors[i + 1];

		JVector<float>& previousLayer = hiddenLayers[i];
		JVector<float>& layerGradient = layerGradients[i + 1];
		JMatrix<float>& weightGradient = weightGradients[i + 1];
		JVector<float>& biasGradient = biasGradients[i + 1];

		for (int row = 0; row < weightMatrix.getRowCount(); row++) {
			float activationDerivative = layerGradient[row];

			for (int col = 0; col < weightMatrix.getColumnCount(); col++) {
				float weightDerivative = previousLayer[col] * activationDerivative;
				weightGradient.addValue(col, row, weightDerivative);
			}

			float biasDerivative = activationDerivative;
			biasGradient.addValue(row, biasDerivative);
		}
	}

	// Compute partial derivatives between last Hidden Layer and Output Layer
	for (int row = 0; row < weightMatrices[hiddenLayerCount].getRowCount(); row++) {
		float activationDerivative = layerGradients[hiddenLayerCount][row];

		for (int col = 0; col < weightMatrices[hiddenLayerCount].getColumnCount(); col++) {
			float weightDerivative = hiddenLayers[hiddenLayerCount - 1][col] * activationDerivative;
			weightGradients[hiddenLayerCount].addValue(col, row, weightDerivative);
		}

		float biasDerivative = activationDerivative;
		biasGradients[hiddenLayerCount].addValue(row, biasDerivative);
	}
}

// Apply gradient vector to current weights and clear stored gradient vector
void NeuralNet::applyGradients(float learningRate, int batchSize) {
	for (int i = 0; i < hiddenLayerCount + 1; i++) {
		JMatrix<float>& weightMatrix = weightMatrices[i];
		JVector<float>& biasVector = biasVectors[i];

		JMatrix<float>& weightGradient = weightGradients[i];
		JVector<float>& biasGradient = biasGradients[i];

		float scalar = -(learningRate / batchSize);

		weightMatrix.addOn(weightGradient.scale(scalar));
		biasVector.addOn(biasGradient.scale(scalar));

		weightGradient.setAllValues(0);
		biasGradient.setAllValues(0);
	}
}
#pragma once
#include "JMatrix.h"

class NeuralNet {
private:
	// includes input and output layers
	//int layers = 5;
	//int neuronsPerLayer[5] = { 784, 300, 200, 100, 10 };
	int hiddenLayerCount = 3;
	//int* neuronsPerLayer;

	JMatrix<float>* weightMatrices;
	JVector<float>* biasVectors;

	JVector<float> inputLayer;
	JVector<float> outputLayer;
	JVector<float>* hiddenLayers;

	//JVector<float>* preActivationLayers;
	JVector<float>* layerGradients;

	JMatrix<float>* weightGradients;
	JVector<float>* biasGradients;

	// distance from 0
	float initValueRange = 1.f;

public:
	NeuralNet() {}

	NeuralNet(int layerCount, ...) {
		int* neuronsPerLayer = new int[layerCount];
		for (int i = 0; i < layerCount; i++) {
			neuronsPerLayer[i] = *(&layerCount + 2 + i * 2);
		}

		if (layerCount < 2) {
			throw "Not enough args for input and output layers";
		}

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
				float randValue = ((rand() / (float)RAND_MAX) * initValueRange * 2) - initValueRange;

				biasVectors[i].setValue(index, randValue);
			}
		}

		//neuronLayers = new JVector<float>[layers];
		//for (int i = 0; i < layers; i++) {
		//	neuronLayers[i] = JVector<float>(neuronsPerLayer[i]);
		//}

		//preActivationLayers = new JVector<float>[hiddenLayerCount + 1];
		layerGradients = new JVector<float>[hiddenLayerCount + 1];

		weightGradients = new JMatrix<float>[hiddenLayerCount + 1];
		biasGradients = new JVector<float>[hiddenLayerCount + 1];

		for (int i = 0; i < hiddenLayerCount + 1; i++) {
			int inputVectorLength = neuronsPerLayer[i];
			int outputVectorLength = neuronsPerLayer[i + 1];

			weightGradients[i] = JMatrix<float>(inputVectorLength, outputVectorLength);
			biasGradients[i] = JVector<float>(outputVectorLength);

			weightGradients[i].setAllValues(0);
			biasGradients[i].setAllValues(0);
		}
	}

	~NeuralNet() {
		delete[] weightMatrices;
		delete[] biasVectors;

		delete[] hiddenLayers;

		//delete[] preActivationLayers;
		delete[] layerGradients;

		delete[] weightGradients;
		delete[] biasGradients;
	}

	NeuralNet(const NeuralNet& copy) {
		//layers = copy.layers;
		// Add later after implementing dynamic neural net structure
		//neuronsPerLayer = copy.neuronsPerLayer
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

		//neuronLayers = new JVector<float>[layers];
		//for (int i = 0; i < layers; i++) {
		//	neuronLayers[i] = copy.neuronLayers[i];
		//}
		//inputLayer = copy.inputLayer;
		//outputLayer = copy.outputLayer;

		
		//preActivationLayers = new JVector<float>[hiddenLayerCount + 1];
		layerGradients = new JVector<float>[hiddenLayerCount + 1];

		weightGradients = new JMatrix<float>[hiddenLayerCount + 1];
		biasGradients = new JVector<float>[hiddenLayerCount + 1];
		for (int i = 0; i < hiddenLayerCount + 1; i++) {
			weightGradients[i] = copy.weightGradients[i];
			biasGradients[i] = copy.biasGradients[i];
		}

		initValueRange = copy.initValueRange;
	}

	const NeuralNet& operator=(const NeuralNet& copy) {
		delete[] weightMatrices;
		delete[] biasVectors;

		delete[] hiddenLayers;

		//delete[] preActivationLayers;
		delete[] layerGradients;

		delete[] weightGradients;
		delete[] biasGradients;

		//layers = copy.layers;
		// Add later after implementing dynamic neural net structure
		//neuronsPerLayer = copy.neuronsPerLayer
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

		//neuronLayers = new JVector<float>[layers];
		//for (int i = 0; i < layers; i++) {
		//	neuronLayers[i] = copy.neuronLayers[i];
		//}
		//inputLayer = copy.inputLayer;
		//outputLayer = copy.outputLayer;

		//preActivationLayers = new JVector<float>[hiddenLayerCount + 1];
		layerGradients = new JVector<float>[hiddenLayerCount + 1];

		weightGradients = new JMatrix<float>[hiddenLayerCount + 1];
		biasGradients = new JVector<float>[hiddenLayerCount + 1];
		for (int i = 0; i < hiddenLayerCount + 1; i++) {
			weightGradients[i] = copy.weightGradients[i];
			biasGradients[i] = copy.biasGradients[i];
		}

		initValueRange = copy.initValueRange;
	}

	float sigmoidFunction(float value) {
		float e = 2.71828;

		return 1 / (1 + pow(e, -value));
	}

	float derivativeSigmoid(float input) {
		float e = 2.71828;

		return 1 / (2 + pow(e, input) + pow(e, -input));
	}

	float derivativePostSigmoid(float input) {
		return input * (1 - input);
	}

	float derivativeCost(float givenValue, float desiredValue) {
		return (givenValue - desiredValue);
	}

	int getLayerCount() {
		return hiddenLayerCount + 2;
	}

	JMatrix<float>& getWeightMatrix(int index) {
		if (index < 0 || index >= hiddenLayerCount + 1) {
			throw "out of bounds";
		}

		return weightMatrices[index];
	}

	JVector<float>& getBiasVector(int index) {
		if (index < 0 || index >= hiddenLayerCount + 1) {
			throw "out of bounds";
		}

		return biasVectors[index];
	}

	//JVector<float>& getPreActivation(int index) {
	//	if (index < 0 || index >= hiddenLayerCount + 1) {
	//		throw "out of bounds";
	//	}

	//	return preActivationLayers[index];
	//}

	JVector<float>& getInputLayer() {
		return inputLayer;
	}

	JVector<float>& getOutputLayer() {
		return outputLayer;
	}

	JVector<float>& getHiddenLayer(int index) {
		if (index < 0 || index >= hiddenLayerCount) {
			throw "out of bounds";
		}

		return hiddenLayers[index];
	}

	void run() {
		// For special case where there are no hidden layers
		if (hiddenLayerCount == 0) {
			//preActivationLayers[0] = biasVectors[0].add(weightMatrices[0].multiply(inputLayer));
			JVector<float> vector = biasVectors[0].add(weightMatrices[0].multiply(inputLayer));

			for (int i = 0; i < vector.getSize(); i++) {
				float sigmoidValue = sigmoidFunction(vector[i]);
				outputLayer.setValue(i, sigmoidValue);
			}

			return;
		}

		// Compute Input Layer into first Hidden layer
		//preActivationLayers[0] = biasVectors[0].add(weightMatrices[0].multiply(inputLayer));
		JVector<float> vectorA = biasVectors[0].add(weightMatrices[0].multiply(inputLayer));
		for (int i = 0; i < vectorA.getSize(); i++) {
			float sigmoidValue = sigmoidFunction(vectorA[i]);
			hiddenLayers[0].setValue(i, sigmoidValue);
		}

		// Compute Hidden Layers
		for (int i = 0; i < hiddenLayerCount - 1; i++) {
			JVector<float> vector = hiddenLayers[i];

			//vector = weightMatrices[i].multiply(vector);
			//vector = biasVectors[i].add(vector);

			//preActivationLayers[i + 1] = biasVectors[i + 1].add(weightMatrices[i + 1].multiply(vector));
			vector = biasVectors[i + 1].add(weightMatrices[i + 1].multiply(vector));

			for (int n = 0; n < vector.getSize(); n++) {
				float sigmoidValue = sigmoidFunction(vector[n]);
				hiddenLayers[i + 1].setValue(n, sigmoidValue);
			}

			//hiddenLayers[i + 1] = vector;
		}

		// Compute last Hidden Layer into Output Layer
		JVector<float> vectorB = biasVectors[hiddenLayerCount].add(weightMatrices[hiddenLayerCount].multiply(hiddenLayers[hiddenLayerCount - 1]));
		for (int i = 0; i < vectorB.getSize(); i++) {
			float sigmoidValue = sigmoidFunction(vectorB[i]);
			outputLayer.setValue(i, sigmoidValue);
		}
	}

	void train(int expectedValue) {
		// Calculate deltas at output layer
		layerGradients[hiddenLayerCount] = outputLayer;
		for (int i = 0; i < outputLayer.getSize(); i++) {
			// Index of output layer matches its corresponding value (0-9)
			float desiredConfidenceValue = 0;
			if (i == expectedValue) {
				desiredConfidenceValue = 1;
			}

			float outputNeuronValue = getOutputLayer()[i];
			float derivativeCostValue = derivativeCost(outputNeuronValue, desiredConfidenceValue);

			//float preActivationSum = getPreActivation(hiddenLayerCount)[i];
			//float derivativeSigmoidValue = derivativeSigmoid(preActivationSum);
			float derivativeSigmoidValue = derivativePostSigmoid(outputNeuronValue);
			float value = derivativeSigmoidValue * derivativeCostValue;
			layerGradients[hiddenLayerCount][i] = value;
		}

		// Calculating deltas at hidden layers
		for (int i = 0; i < hiddenLayerCount; i++) {
			//int deltaIndex = layers - 3 - i;
			int deltaIndex = hiddenLayerCount - 1 - i;
			JMatrix<float>& weightMatrix = getWeightMatrix(deltaIndex + 1);
			//JMatrix<float> transposedMatrix = weightMatrix.transpose();
			//layerGradients[deltaIndex] = transposedMatrix.multiply(layerGradients[deltaIndex + 1]);
			layerGradients[deltaIndex] = weightMatrix.transposedMultiply(layerGradients[deltaIndex + 1]);

			for (int index = 0; index < layerGradients[deltaIndex].getSize(); index++) {
				//float preActivationSum = getPreActivation(deltaIndex)[index];
				//float derivativeSigmoidValue = derivativeSigmoid(preActivationSum);
				float postActivationValue = hiddenLayers[deltaIndex][index];
				float derivativeSigmoidValue = derivativePostSigmoid(postActivationValue);
				float value = layerGradients[deltaIndex][index] * derivativeSigmoidValue;
				layerGradients[deltaIndex][index] = value;
			}
		}

		//// Computing gradient vectors for weights and biases
		//for (int i = 0; i < hiddenLayerCount + 1; i++) {
		//	JMatrix<float>& weightMatrix = getWeightMatrix(i);
		//	JVector<float>& biasVector = getBiasVector(i);

		//	JVector<float>& previousLayer = getNeuronLayer(i);
		//	
		//	JVector<float>& delta = layerGradients[i];

		//	JMatrix<float>& weightGradient = weightGradients[i];
		//	JVector<float>& biasGradient = biasGradients[i];

		//	for (int row = 0; row < weightMatrix.getRowCount(); row++) {
		//		float activationDerivative = delta[row];

		//		for (int col = 0; col < weightMatrix.getColumnCount(); col++) {
		//			float weightDerivative = previousLayer[col] * activationDerivative;
		//			weightGradient.addValue(col, row, weightDerivative);
		//		}

		//		float biasDerivative = activationDerivative;
		//		biasGradient.addValue(row, biasDerivative);
		//	}
		//}

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
			JMatrix<float>& weightMatrix = getWeightMatrix(i + 1);
			JVector<float>& biasVector = getBiasVector(i + 1);

			JVector<float>& previousLayer = hiddenLayers[i];

			JVector<float>& layerGradient = layerGradients[i + 1];

			JMatrix<float>& weightGradient = weightGradients[i + 1];
			JVector<float>& biasGradient = biasGradients[i + 1];

			for (int row = 0; row < weightMatrix.getRowCount(); row++) {
				float activationDerivative = layerGradient[row];

				for (int col = 0; col < weightMatrix.getColumnCount(); col++) {
					float weightDerivative = inputLayer[col] * activationDerivative;
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
				float weightDerivative = inputLayer[col] * activationDerivative;
				weightGradients[hiddenLayerCount].addValue(col, row, weightDerivative);
			}

			float biasDerivative = activationDerivative;
			biasGradients[hiddenLayerCount].addValue(row, biasDerivative);
		}
	}

	// Apply gradient vector to current weights and clear stored gradient vector
	void applyGradients(float learningRate, int batchSize) {
		for (int i = 0; i < hiddenLayerCount + 1; i++) {
			JMatrix<float>& weightMatrix = weightMatrices[i];
			JVector<float>& biasVector = biasVectors[i];

			JMatrix<float>& weightGradient = weightGradients[i];
			JVector<float>& biasGradient = biasGradients[i];

			weightGradient.scale(-learningRate / batchSize);
			biasGradient.scale(-learningRate / batchSize);

			weightMatrix.addOn(weightGradient);
			biasVector.addOn(biasGradient);

			weightGradient.setAllValues(0);
			biasGradient.setAllValues(0);
		}
	}
};
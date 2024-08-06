#pragma once
#include "JMatrix.h"

class NeuralNet {
private:
	// includes input and output layers
	int layers = 5;
	int neuronsPerLayer[5] = { 784, 300, 200, 100, 10 };

	JMatrix<float>* weightMatrices;
	JVector<float>* biasVectors;

	JVector<float>* neuronLayers;

	JVector<float>* preActivationLayers;
	JVector<float>* layerGradients;

	JMatrix<float>* weightGradients;
	JVector<float>* biasGradients;

	// distance from 0
	float initValueRange = 1.f;

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

		neuronLayers = new JVector<float>[layers];
		for (int i = 0; i < layers; i++) {
			neuronLayers[i] = JVector<float>(neuronsPerLayer[i]);
		}

		preActivationLayers = new JVector<float>[layers - 1];
		layerGradients = new JVector<float>[layers - 1];

		weightGradients = new JMatrix<float>[layers - 1];
		biasGradients = new JVector<float>[layers - 1];

		for (int i = 0; i < layers - 1; i++) {
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

		delete[] neuronLayers;

		delete[] preActivationLayers;
		delete[] layerGradients;

		delete[] weightGradients;
		delete[] biasGradients;
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

		neuronLayers = new JVector<float>[layers];
		for (int i = 0; i < layers; i++) {
			neuronLayers[i] = copy.neuronLayers[i];
		}
		
		preActivationLayers = new JVector<float>[layers - 1];
		layerGradients = new JVector<float>[layers - 1];

		weightGradients = new JMatrix<float>[layers - 1];
		biasGradients = new JVector<float>[layers - 1];
		for (int i = 0; i < layers - 1; i++) {
			weightGradients[i] = copy.weightGradients[i];
			biasGradients[i] = copy.biasGradients[i];
		}

		initValueRange = copy.initValueRange;
	}

	const NeuralNet& operator=(const NeuralNet& copy) {
		delete[] weightMatrices;
		delete[] biasVectors;

		delete[] neuronLayers;

		delete[] preActivationLayers;
		delete[] layerGradients;

		delete[] weightGradients;
		delete[] biasGradients;

		layers = copy.layers;
		// Add later after implementing dynamic neural net structure
		//neuronsPerLayer = copy.neuronsPerLayer

		weightMatrices = new JMatrix<float>[layers - 1];
		biasVectors = new JVector<float>[layers - 1];

		for (int i = 0; i < layers - 1; i++) {
			weightMatrices[i] = copy.weightMatrices[i];
			biasVectors[i] = copy.biasVectors[i];
		}

		neuronLayers = new JVector<float>[layers];
		for (int i = 0; i < layers; i++) {
			neuronLayers[i] = copy.neuronLayers[i];
		}
		
		preActivationLayers = new JVector<float>[layers - 1];
		layerGradients = new JVector<float>[layers - 1];

		weightGradients = new JMatrix<float>[layers - 1];
		biasGradients = new JVector<float>[layers - 1];

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

	float derivativeCost(float givenValue, float desiredValue) {
		return (givenValue - desiredValue);
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

		return preActivationLayers[index];
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

			preActivationLayers[i] = vector;

			for (int n = 0; n < neuronsPerLayer[i + 1]; n++) {
				float sigmoidValue = sigmoidFunction(vector[n]);
				vector.setValue(n, sigmoidValue);
			}

			neuronLayers[i + 1] = vector;
		}
	}

	void train(int expectedValue) {
		// Calculate deltas at output layer
		layerGradients[layers - 2] = getOutputLayer();
		for (int i = 0; i < getOutputLayer().getSize(); i++) {
			// Index of output layer matches its corresponding value (0-9)
			float desiredConfidenceValue = 0;
			if (i == expectedValue) {
				desiredConfidenceValue = 1;
			}

			float outputNeuronValue = getOutputLayer()[i];
			float derivativeCostValue = derivativeCost(outputNeuronValue, desiredConfidenceValue);

			float preActivationSum = getPreActivation(layers - 2)[i];
			float derivativeSigmoidValue = derivativeSigmoid(preActivationSum);
			float value = derivativeSigmoidValue * derivativeCostValue;
			layerGradients[layers - 2][i] = value;
		}

		// Calculating deltas at hidden layers
		for (int i = 0; i < layers - 2; i++) {
			int deltaIndex = layers - 3 - i;
			JMatrix<float>& weightMatrix = getWeightMatrix(deltaIndex + 1);
			//JMatrix<float> transposedMatrix = weightMatrix.transpose();
			//layerGradients[deltaIndex] = transposedMatrix.multiply(layerGradients[deltaIndex + 1]);
			layerGradients[deltaIndex] = weightMatrix.transposedMultiply(layerGradients[deltaIndex + 1]);

			for (int index = 0; index < layerGradients[deltaIndex].getSize(); index++) {
				float preActivationSum = getPreActivation(deltaIndex)[index];
				float derivativeSigmoidValue = derivativeSigmoid(preActivationSum);
				float value = layerGradients[deltaIndex][index] * derivativeSigmoidValue;
				layerGradients[deltaIndex][index] = value;
			}
		}

		// Adjusting weights and biases
		for (int i = 0; i < layers - 1; i++) {
			JMatrix<float>& weightMatrix = getWeightMatrix(i);
			JVector<float>& biasVector = getBiasVector(i);

			JVector<float>& previousLayer = getNeuronLayer(i);
			JVector<float>& delta = layerGradients[i];

			JMatrix<float>& weightGradient = weightGradients[i];
			JVector<float>& biasGradient = biasGradients[i];

			for (int row = 0; row < weightMatrix.getRowCount(); row++) {
				float activationDerivative = delta[row];

				for (int col = 0; col < weightMatrix.getColumnCount(); col++) {
					float weightDerivative = previousLayer[col] * activationDerivative;
					//float newWeightDerivative = weightGradient.getValue(col, row) + weightDerivative;
					//weightGradient.setValue(col, row, newWeightDerivative);
					weightGradient.addValue(col, row, weightDerivative);
					//float newWeight = weightGradient.getValue(col, row) - ((weightDerivative * learningRate) / (float)batchSize);
					//weightMatrix.setValue(col, row, newWeight);
				}

				float biasDerivative = activationDerivative;
				//float newBiasDerivative = biasGradient[row] + biasDerivative;
				//biasGradient.setValue(row, newBiasDerivative);
				biasGradient.addValue(row, biasDerivative);
				//float newBias = biasGradient[row] - ((biasDerivative * learningRate) / (float)batchSize);
				//biasVector.setValue(row, newBias);
			}
		}
	}

	void applyGradients(float learningRate, int batchSize) {
		for (int i = 0; i < layers - 1; i++) {
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
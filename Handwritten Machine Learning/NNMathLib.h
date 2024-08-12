#pragma once
#include <math.h>
#include "JVector.h"

static JVector<float> makeHotVector(int count, int desiredValue) {
    JVector<float> vec(count);
    for (int i = 0; i < count; i++) {
        vec[i] = 0;
    }
    vec[desiredValue] = 1;

    return vec;
}

// Sigmoid Functions
static float sigmoidFunction(float value) { return 1 / (1 + exp(-value)); }
static void applySigmoid(JVector<float>& vector) { for (int i = 0; i < vector.getSize(); i++) { vector[i] = sigmoidFunction(vector[i]); } }
static float derivativePostSigmoid(float input) { return input * (1 - input); }

// Softmax and Cross Entropy Loss Functions
static void applySoftmax(JVector<float>& vector) {
	float sum = 0;
	for (int i = 0; i < vector.getSize(); i++) {
		vector[i] = exp(vector[i]);
		sum += vector[i];
	}

	vector.scale(1 / sum);
}

static float crossEntropyLoss(float desiredValue, float givenValue) {
    return -desiredValue * log(givenValue);
}

static float computeCrossEntropy(const JVector<float>& desiredVector, const JVector<float>& givenVector) {
    float sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += crossEntropyLoss(desiredVector[i], givenVector[i]);
    }

    return sum;
}

// Calculates Softmax and Cross Entropy derivatives in one
static float derivativeSoftmaxCrossEntropy(float desiredValue, float givenValue) {
    return givenValue - desiredValue;
}

static void applyDerivativeSoftmaxCrossEntropy(const JVector<float>& desiredVector, JVector<float>& givenVector) {
    for (int i = 0; i < givenVector.getSize(); i++) {
        givenVector[i] = derivativeSoftmaxCrossEntropy(desiredVector[i], givenVector[i]);
    }
}

// Square Loss Functions
static float squareLoss(float desiredValue, float givenValue) {
    float difference = desiredValue - givenValue;
    // multiply by half so the derivative doesn't have a coefficient
    return difference * difference * 0.5;
}

static float computeMeanSquareError(const JVector<float>& desiredVector, const JVector<float>& givenVector) {
    float sum = 0;
    for (int i = 0; i < desiredVector.getSize(); i++) {
        sum += squareLoss(desiredVector[i], givenVector[i]);
    }

    return sum / desiredVector.getSize();
}
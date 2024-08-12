#pragma once
#include <iostream>
#include "NeuralNet.h"

void saveNeuralNet(NeuralNet& neuralNet) {
    std::ofstream fileStream;
    fileStream.open("nn-saves/nn-data.idx3-ubyte", std::ios::binary | std::ios::trunc);

    std::cout << "file ";
    if (fileStream.is_open()) {
        std::cout << "successfully opened.";
    }
    else {
        std::cout << "failed to open.";
    }
    std::cout << std::endl;

    int layers = neuralNet.getLayerCount();
    fileStream.write((char*)&layers, sizeof(layers));

    int inputLayerSize = neuralNet.getInputLayer().getSize();
    fileStream.write((char*)&inputLayerSize, sizeof(inputLayerSize));

    for (int i = 0; i < layers - 2; i++) {
        int hiddenLayerSize = neuralNet.getHiddenLayer(i).getSize();
        fileStream.write((char*)&hiddenLayerSize, sizeof(hiddenLayerSize));
    }

    int outputLayerSize = neuralNet.getOutputLayer().getSize();
    fileStream.write((char*)&outputLayerSize, sizeof(outputLayerSize));

    for (int i = 0; i < layers - 1; i++) {
        JMatrix<float>& weightMatrix = neuralNet.getWeightMatrix(i);
        JVector<float>& biasVector = neuralNet.getBiasVector(i);

        fileStream.write((char*)weightMatrix.getDataPtr(), sizeof(float) * weightMatrix.getColumnCount() * weightMatrix.getRowCount());
        fileStream.write((char*)biasVector.getDataPtr(), sizeof(float) * biasVector.getSize());
    }

    fileStream.close();
}

void loadNeuralNet(NeuralNet& neuralNet) {
    std::ifstream fileStream;
    fileStream.open("nn-saves/nn-data.idx3-ubyte", std::ios::binary);

    std::cout << "file ";
    if (fileStream.is_open()) {
        std::cout << "successfully opened.";
    }
    else {
        std::cout << "failed to open.";
    }
    std::cout << std::endl;

    int layers;
    fileStream.read((char*)&layers, sizeof(layers));

    if (layers < 2) {
        throw "not enough neural net layers";
    }

    int* neuronsPerLayer = new int[layers];
    fileStream.read((char*)neuronsPerLayer, sizeof(int) * layers);

    neuralNet = NeuralNet(layers, neuronsPerLayer);

    for (int i = 0; i < layers - 1; i++) {
        JMatrix<float>& weightMatrix = neuralNet.getWeightMatrix(i);
        JVector<float>& biasVector = neuralNet.getBiasVector(i);

        fileStream.read((char*)weightMatrix.getDataPtr(), sizeof(float) * weightMatrix.getColumnCount() * weightMatrix.getRowCount());
        fileStream.read((char*)biasVector.getDataPtr(), sizeof(float) * biasVector.getSize());
    }

    fileStream.close();
}
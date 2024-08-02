#include <string>
#include "raylib.h"
#include "rlgl.h"
#include "PixelGrid.h"
#include "MnistParser.h"
#include "NeuralNet.h"
#include <iostream>

float costFunction(float givenValue, float desiredValue) {
    float difference = givenValue - desiredValue;
    return difference * difference;
}

float computeCostSum(int expectedValue, JVector<float> confidenceValues) {
    float cost = 0;

    for (int i = 0; i < 10; i++) {
        float desiredConfidenceLevel = 0;

        if (i == expectedValue) {
            desiredConfidenceLevel = 1;
        }

        cost += costFunction(confidenceValues[i], desiredConfidenceLevel);
    }

    return cost;
}

float derivativeCost(float givenValue, float desiredValue) {
    return 2 * (givenValue - desiredValue);
}

float derivativeSigmoid(float input) {
    float e = 2.71828;

    return 1 / (2 + pow(e, input) + pow(e, -input));
}

// TODO:
// 1. add bias vector training (somehow I forgot about them)
// 2. calculate deltas beforehand (rename error vector to delta)
//
//

void train(NeuralNet& neuralNet, int expectedValue, float learningRate) {
    JVector<float>& outputLayer = neuralNet.getOutputLayer();

    int layers = neuralNet.getLayerCount();

    JMatrix<float>* deltaErrors = new JMatrix<float>[layers - 1];
    deltaErrors[layers - 2] = JMatrix<float>(1, neuralNet.getOutputLayer().getSize());

    // Calculate deltas at output layer
    for (int i = 0; i < deltaErrors[layers - 2].getRowCount(); i++) {
        // Index of output layer matches its corresponding value (0-9)
        float desiredConfidenceValue = 0;
        if (i == expectedValue) {
            desiredConfidenceValue = 1;
        }

        float outputNeuronValue = outputLayer[i];
        float derivativeCostValue = derivativeCost(outputNeuronValue, desiredConfidenceValue);
        float derivativeSigmoidValue = derivativeSigmoid(derivativeCostValue);
        float value = derivativeSigmoidValue * derivativeCostValue;
        deltaErrors[layers - 2].setValue(0, i, value);
    }

    // Calculating deltas at hidden layers
    for (int i = 0; i < layers - 2; i++) {
        int deltaErrorIndex = layers - 3 - i;
        JMatrix<float>& weightMatrix = neuralNet.getWeightMatrix(deltaErrorIndex + 1);
        JMatrix<float> transposedMatrix = weightMatrix.transpose();
        deltaErrors[deltaErrorIndex] = transposedMatrix.multiply(deltaErrors[deltaErrorIndex + 1]);

        for (int i = 0; i < deltaErrors[deltaErrorIndex].getColumnCount(); i++) {
            float derivativeSigmoidValue = derivativeSigmoid(deltaErrors[deltaErrorIndex].getValue(0, i));
            deltaErrors[deltaErrorIndex].setValue(0, i, derivativeSigmoidValue);
        }
    }

    // Adjusting weights and biases
    for (int i = 0; i < layers - 1; i++) {
        JMatrix<float>& weightMatrix = neuralNet.getWeightMatrix(i);
        JVector<float>& biasVector = neuralNet.getBiasVector(i);
        JVector<float>& previousLayer = neuralNet.getNeuronLayer(i);
        JMatrix<float>& deltaError = deltaErrors[i];

        for (int row = 0; row < weightMatrix.getRowCount(); row++) {
            //float currentNeuronError = errorVector.getValue(0, row);
            float currentNeuronError = deltaError.getValue(0, row);

            for (int col = 0; col < weightMatrix.getColumnCount(); col++) {
                float weightGradient = previousLayer[col] * currentNeuronError;
                float newWeight = weightMatrix.getValue(col, row) - (weightGradient * learningRate);
                weightMatrix.setValue(col, row, newWeight);
            }

            float biasGradient = currentNeuronError;
            float newBias = biasVector[row] - (biasGradient * learningRate);
            biasVector.setValue(row, newBias);
        }
    }

    delete[] deltaErrors;
}

void loadGridFromDataset(PixelGrid& pixelGrid, MnistParser& dataset, int imageIndex) {
    int pixelCount = dataset.getColumnCount() * dataset.getRowCount();
    int offset = imageIndex * pixelCount;
    std::memcpy(pixelGrid.getDataPtr(), dataset.getImageBuffer() + offset, sizeof(byte) * pixelCount);
}

void loadGridValuesIntoNN(NeuralNet& neuralNet, PixelGrid& pixelGrid) {
    JVector<float>& inputLayer = neuralNet.getInputLayer();
    int pixelCount = inputLayer.getSize();

    for (int i = 0; i < pixelCount; i++) {
        byte byteValue = pixelGrid.getDataPtr()[i];
        inputLayer.setValue(i, (float)byteValue);
    }
}

int main() {
    srand(32498784);

    int screenWidth = 1600;
    int screenHeight = 800;

    InitWindow(screenWidth, screenHeight, "Handwritten Machine Learning");

    SetTargetFPS(240);

    bool trainingMode = false;

    std::string imagesFileName;
    std::string labelsFileName;

    if (trainingMode) {
        imagesFileName = "datasets/train-images.idx3-ubyte";
        labelsFileName = "datasets/train-labels.idx1-ubyte";
    }
    else {
        imagesFileName = "datasets/t10k-images.idx3-ubyte";
        labelsFileName = "datasets/t10k-labels.idx1-ubyte";
    }

    MnistParser dataset;

    dataset.loadImageBuffer(imagesFileName);
    dataset.loadLabelBuffer(labelsFileName);
    
    PixelGrid grid({ 100, 100 }, 600, 28);
    NeuralNet neuralNet = NeuralNet();
    
    //for (int index = 0; index < 1; index++) {
    //    std::cout << "generation: " + std::to_string(index) << std::endl;
    //    for (int i = 0; i < 100; i++) {
    //        std::cout << "iteration: " + std::to_string(i) << std::endl;
    //        int expectedValue = dataset.getLabelBuffer()[index];
    //        loadGridFromDataset(grid, dataset, index);
    //        loadGridValuesIntoNN(neuralNet, grid);
    //        neuralNet.run();
    //        train(neuralNet, expectedValue, 0.01f / (i + 1));
    //    }
    //}

    loadGridFromDataset(grid, dataset, 0);
    loadGridValuesIntoNN(neuralNet, grid);
    neuralNet.run();

    int expectedValue;
    float evaluatedCost;

    int currentImageIndex = 0;

    bool training = false;
    int epochs = 0;
    int iterations = 0;

    while (!WindowShouldClose()) {
        // Updates
        float delta = GetFrameTime();
        Vector2 mousePos = GetMousePosition();
        
        std::pair<int, int> cellCoords = grid.getCellCoords(mousePos);
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            if (cellCoords != std::pair<int, int>(-1, -1)) {
                grid.setCellValue(cellCoords.first, cellCoords.second, 255);

                loadGridValuesIntoNN(neuralNet, grid);
                neuralNet.run();
            }
        }

        if (IsKeyPressed(KEY_RIGHT) && currentImageIndex < dataset.getImageCount() - 1) {
            currentImageIndex++;
            loadGridFromDataset(grid, dataset, currentImageIndex);
            
            loadGridValuesIntoNN(neuralNet, grid);
            neuralNet.run();
        }
        else if (IsKeyPressed(KEY_LEFT) && currentImageIndex > 0) {
            currentImageIndex--;
            loadGridFromDataset(grid, dataset, currentImageIndex);
            
            loadGridValuesIntoNN(neuralNet, grid);
            neuralNet.run();
        }

        if (IsKeyPressed(KEY_SPACE)) {
            grid.invertBlackWhite();
        }


        expectedValue = dataset.getLabelBuffer()[currentImageIndex];

        if (IsKeyPressed(KEY_T) && !training) {
            training = true;
            iterations = 0;
        }

        if (training) {
            int expected = dataset.getLabelBuffer()[iterations];
            loadGridFromDataset(grid, dataset, iterations);
            loadGridValuesIntoNN(neuralNet, grid);
            neuralNet.run();
            train(neuralNet, expected, 0.0001f);

            iterations++;
        }

        if (iterations >= 1000) {
            training = false;
            iterations = 0;
            epochs++;
        }


        evaluatedCost = computeCostSum(expectedValue, neuralNet.getOutputLayer());
        
        // Drawing
        BeginDrawing();

        ClearBackground(RAYWHITE);

        grid.draw();

        std::string cell = std::to_string(cellCoords.first) + ", " + std::to_string(cellCoords.second);
        DrawText(cell.c_str(), 10, 40, 20, GREEN);

        std::string imageIndexStr = "Image Index: " + std::to_string(currentImageIndex);
        DrawText(imageIndexStr.c_str(), 200, 20, 20, BLUE);

        std::string expectedValueStr = "Expected Value: " + std::to_string(expectedValue);
        DrawText(expectedValueStr.c_str(), 1200, 120, 20, BLUE);

        std::string costStr = "Evaluated Cost: " + std::to_string(evaluatedCost);
        DrawText(costStr.c_str(), 1200, 160, 20, BLUE);

        std::string trainingData = "Training Info:\n\nEpochs: " + std::to_string(epochs) + "\n\nIterations: " + std::to_string(iterations);
        DrawText(trainingData.c_str(), 1000, 200, 20, RED);

        DrawText("Confidence Values:", 800, 100, 20, BLUE);
        for (int i = 0; i < 10; i++) {
            float value = neuralNet.getOutputLayer()[i];
            std::string str = std::to_string(i) + " " + std::to_string(value);

            DrawText(str.c_str(), 800, 140 + 30 * i, 20, BLUE);
        }

        DrawFPS(10, 10);

        EndDrawing();
    }
}

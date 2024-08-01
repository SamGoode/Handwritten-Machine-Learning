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

float computeCostSum(int expectedValue, JMatrix<float> confidenceValues) {
    float cost = 0;

    for (int i = 0; i < 10; i++) {
        float desiredConfidenceLevel = 0;

        if (i == expectedValue) {
            desiredConfidenceLevel = 1;
        }

        cost += costFunction(confidenceValues.getValue(0, i), desiredConfidenceLevel);
    }

    return cost;
}

float derivativeCost(float givenValue, float desiredValue) {
    return 2 * abs(givenValue - desiredValue);
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
    JMatrix<float>& outputLayer = neuralNet.getOutputLayer();

    // Backpropagated error vector
    JMatrix<float> errorVector = JMatrix<float>(1, outputLayer.getRowCount());
    for (int i = 0; i < errorVector.getRowCount(); i++) {
        //index of output layer matches its corresponding value (0-9)
        float desiredConfidenceValue = 0;
        if (i == expectedValue) {
            desiredConfidenceValue = 1;
        }

        float error = derivativeCost(outputLayer.getValue(0, i), desiredConfidenceValue);
        errorVector.setValue(0, i, error);
    }

    // 3 2 1 0
    int layers = neuralNet.getLayerCount();

    JMatrix<float>* deltaErrors = new JMatrix<float>[layers - 1];


    deltaErrors[layers - 2] = errorVector;
    // 3 2 1
    for (int matrixIndex = layers - 2; matrixIndex > 0; matrixIndex--) {

        for (int i = 0; i < deltaErrors[matrixIndex].getRowCount(); i++) {
            float value = derivativeSigmoid(deltaErrors[matrixIndex].getValue(0, i));
            deltaErrors[matrixIndex].setValue(0, i, value);
        }

        JMatrix<float>& weightMatrix = neuralNet.getWeightMatrix(matrixIndex);

        JMatrix<float> weightMatrixT = weightMatrix.transpose();
        deltaErrors[matrixIndex - 1] = weightMatrixT.multiply(deltaErrors[matrixIndex]);
    }
    for (int i = 0; i < deltaErrors[0].getRowCount(); i++) {
        float value = derivativeSigmoid(deltaErrors[0].getValue(0, i));
        deltaErrors[0].setValue(0, i, value);
    }

    for (int matrixIndex = layers - 2; matrixIndex >= 0; matrixIndex--) {
        //for (int i = 0; i < errorVector.getRowCount(); i++) {
        //    float value = derivativeSigmoid(errorVector.getValue(0, i));
        //    errorVector.setValue(0, i, value);
        //}

        JMatrix<float>& weightMatrix = neuralNet.getWeightMatrix(matrixIndex);
        JMatrix<float>& biasVector = neuralNet.getBiasVector(matrixIndex);
        

        JMatrix<float>& previousLayer = neuralNet.getNeuronLayer(matrixIndex);
        for (int row = 0; row < weightMatrix.getRowCount(); row++) {
            //float currentNeuronError = errorVector.getValue(0, row);
            float currentNeuronError = deltaErrors[matrixIndex].getValue(0, row);

            for (int col = 0; col < weightMatrix.getColumnCount(); col++) {
                float weightChange = -(previousLayer.getValue(0, col) * currentNeuronError * learningRate);
                float newWeight = weightMatrix.getValue(col, row) + weightChange;
                weightMatrix.setValue(col, row, newWeight);
            }

            float biasChange = -(currentNeuronError * learningRate);
            float newBias = biasVector.getValue(0, row) + biasChange;
            biasVector.setValue(0, row, newBias);
        }

        //JMatrix<float> weightMatrixT = weightMatrix.transpose();
        //errorVector = weightMatrixT.multiply(errorVector);
    }
}

void loadGridFromDataset(PixelGrid& pixelGrid, MnistParser& dataset, int imageIndex) {
    int pixelCount = dataset.getColumnCount() * dataset.getRowCount();
    int offset = imageIndex * pixelCount;
    std::memcpy(pixelGrid.getDataPtr(), dataset.getImageBuffer() + offset, sizeof(byte) * pixelCount);
}

void loadGridValuesIntoNN(NeuralNet& neuralNet, PixelGrid& pixelGrid) {
    JMatrix<float>& inputLayer = neuralNet.getInputLayer();
    int pixelCount = inputLayer.getRowCount();

    for (int i = 0; i < pixelCount; i++) {
        byte byteValue = pixelGrid.getDataPtr()[i];
        inputLayer.setValue(0, i, (float)byteValue);
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
    
    for (int index = 0; index < 10; index++) {
        std::cout << "generation: " + std::to_string(index) << std::endl;
        for (int i = 0; i < 500; i++) {
            std::cout << "iteration: " + std::to_string(i) << std::endl;
            int expectedValue = dataset.getLabelBuffer()[index];
            loadGridFromDataset(grid, dataset, index);
            loadGridValuesIntoNN(neuralNet, grid);
            neuralNet.run();
            train(neuralNet, expectedValue, 0.005f / (i + 1));
        }
    }

    loadGridFromDataset(grid, dataset, 0);
    loadGridValuesIntoNN(neuralNet, grid);
    neuralNet.run();

    int expectedValue;
    float evaluatedCost;

    int currentImageIndex = 0;

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

        DrawText("Confidence Values:", 800, 100, 20, BLUE);
        for (int i = 0; i < 10; i++) {
            float value = neuralNet.getOutputLayer().getValue(0, i);
            std::string str = std::to_string(i) + " " + std::to_string(value);

            DrawText(str.c_str(), 800, 140 + 30 * i, 20, BLUE);
        }

        DrawFPS(10, 10);

        EndDrawing();
    }
}

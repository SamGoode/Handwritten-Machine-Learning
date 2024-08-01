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
    return 2 * (givenValue - desiredValue);
}

float derivativeSigmoid(float input) {
    float e = 2.71828;

    return 1 / (2 + pow(e, input) + pow(e, -input));
}

void train(NeuralNet& neuralNet, int expectedValue) {
    float learningRate = 0.2f;

    JMatrix<float>& outputLayer = neuralNet.getOutputLayer();

    // Backpropagated error vector
    JMatrix<float> errorVector = JMatrix<float>(1, outputLayer.getColumnCount());
    for (int i = 0; i < errorVector.getColumnCount(); i++) {
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
    for (int matrixIndex = layers - 2; matrixIndex >= 0; matrixIndex--) {
        for (int i = 0; i < errorVector.getColumnCount(); i++) {
            float value = derivativeSigmoid(errorVector.getValue(0, i));
            errorVector.setValue(0, i, value);
        }

        JMatrix<float>& weightMatrix = neuralNet.getWeightMatrix(matrixIndex);
        JMatrix<float>& previousLayer = neuralNet.getNeuronLayer(matrixIndex);
        for (int row = 0; row < weightMatrix.getRowCount(); row++) {
            float currentNeuronError = errorVector.getValue(0, row);

            for (int col = 0; col < weightMatrix.getColumnCount(); col++) {
                float newValue = weightMatrix.getValue(col, row) + (previousLayer.getValue(0, col) * currentNeuronError * learningRate);
                weightMatrix.setValue(col, row, newValue);
            }
        }

        JMatrix<float> weightMatrixT = weightMatrix.transpose();
        errorVector = weightMatrixT.multiply(errorVector);
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
    loadGridFromDataset(grid, dataset, 0);

    NeuralNet neuralNet = NeuralNet();
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

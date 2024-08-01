#include <string>
#include "raylib.h"
#include "rlgl.h"
#include "PixelGrid.h"
#include "MnistParser.h"
#include "NeuralNet.h"
#include <iostream>

float costFunction(int expectedValue, JMatrix<float> confidenceValues) {
    float cost = 0;

    for (int i = 0; i < 10; i++) {
        float desiredConfidenceLevel = 0;

        if (i == expectedValue) {
            desiredConfidenceLevel == 1;
        }

        float difference = desiredConfidenceLevel - confidenceValues.getValue(0, i);

        cost += difference * difference;
    }

    return cost;
}

void loadGridFromDataset(PixelGrid& pixelGrid, MnistParser& dataset, int imageIndex) {
    int pixelCount = 784;
    int offset = imageIndex * pixelCount;
    std::memcpy(pixelGrid.getDataPtr(), dataset.getImageBuffer() + offset, sizeof(byte) * pixelCount);
}

JMatrix<float> runNeuralNet(NeuralNet& neuralNet, PixelGrid& pixelGrid) {
    int pixelCount = 784;

    JMatrix<float> inputVector(1, pixelCount);
    for (int i = 0; i < pixelCount; i++) {
        byte byteValue = pixelGrid.getDataPtr()[i];
        inputVector.setValue(0, i, (float)byteValue);
    }

    return neuralNet.computeOutput(inputVector);
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

    // Matrix Class test
    JMatrix<int> mat1(3, 2);
    JMatrix<int> mat2(2, 3);
    int value = 0;
    for (int i = 0; i < 2; i++) {
        for (int n = 0; n < 3; n++) {
            value++;
            mat1.setValue(n, i, value);
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int n = 0; n < 2; n++) {
            value++;
            mat2.setValue(n, i, value);
        }
    }

    std::cout << mat1.toString();
    std::cout << mat2.toString();

    JMatrix<int> resultMat = mat1.multiply(mat2);
    std::cout << resultMat.toString();

    MnistParser dataset;

    dataset.loadImageBuffer(imagesFileName);
    dataset.loadLabelBuffer(labelsFileName);
    
    PixelGrid grid({ 100, 100 }, 600, 28);
    loadGridFromDataset(grid, dataset, 0);

    NeuralNet neuralNet = NeuralNet();
    JMatrix<float> outputVector;
    outputVector = runNeuralNet(neuralNet, grid);
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

                outputVector = runNeuralNet(neuralNet, grid);
            }
        }

        if (IsKeyPressed(KEY_RIGHT) && currentImageIndex < dataset.getImageCount() - 1) {
            currentImageIndex++;
            loadGridFromDataset(grid, dataset, currentImageIndex);
            outputVector = runNeuralNet(neuralNet, grid);
        }
        else if (IsKeyPressed(KEY_LEFT) && currentImageIndex > 0) {
            currentImageIndex--;
            loadGridFromDataset(grid, dataset, currentImageIndex);
            outputVector = runNeuralNet(neuralNet, grid);
        }

        if (IsKeyPressed(KEY_SPACE)) {
            grid.invertBlackWhite();
        }

        expectedValue = dataset.getLabelBuffer()[currentImageIndex];
        evaluatedCost = costFunction(expectedValue, outputVector);
        
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
            float value = outputVector.getValue(0, i);
            std::string str = std::to_string(i) + " " + std::to_string(value);

            DrawText(str.c_str(), 800, 140 + 30 * i, 20, BLUE);
        }

        DrawFPS(10, 10);

        EndDrawing();
    }
}

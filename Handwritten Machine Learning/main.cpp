#include <string>
#include "raylib.h"
#include "rlgl.h"
#include "PixelGrid.h"
#include "MnistParser.h"
#include "NeuralNet.h"
#include <iostream>
#include <algorithm>

//static float costFunction(float givenValue, float desiredValue) {
//    float difference = givenValue - desiredValue;
//    return difference * difference * 0.5; //multiply by half so the derivative doesn't have a coefficient
//}
//
//static float computeMeanError(int expectedValue, const JVector<float>& confidenceValues) {
//    float sum = 0;
//
//    for (int i = 0; i < 10; i++) {
//        float desiredConfidenceLevel = 0;
//
//        if (i == expectedValue) {
//            desiredConfidenceLevel = 1;
//        }
//
//        sum += costFunction(confidenceValues[i], desiredConfidenceLevel);
//    }
//
//    return sum / (10 / 2);
//}

static float costFunction(float givenValue, float desiredValue) {
    return -desiredValue * log(givenValue);
}

static float computeCrossEntropy(int expectedValue, const JVector<float>& confidenceValues) {
    float sum = 0;

    for (int i = 0; i < 10; i++) {
        float desiredConfidenceLevel = 0;

        if (i == expectedValue) {
            desiredConfidenceLevel = 1;
        }

        sum += costFunction(confidenceValues[i], desiredConfidenceLevel);
    }

    return sum;
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

int getHighestIndex(const JVector<float>& vector) {
    int index = 0;
    float maxValue = 0;
    for (int i = 0; i < vector.getSize(); i++) {
        if (maxValue < vector[i]) {
            index = i;
            maxValue = vector[i];
        }
    }

    return index;
}

void saveNeuralNet(NeuralNet& neuralNet, std::string fileName) {
    std::ofstream fileStream;
    fileStream.open("nn-saves/nn-data.idx3-ubyte", std::ios::binary);

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

        fileStream.write((char*)weightMatrix.getDataPtr(), sizeof(float) * weightMatrix.getColumnCount() * weightMatrix.getRowCount());
    }
}

void loadNeuralNet(NeuralNet& neuralNet, std::string fileName) {
    std::ifstream fileStream;
    fileStream.open("nn-saves/nn-data.idx3-ubyte", std::ios::binary);

    int layers;
    fileStream.read((char*)&layers, sizeof(layers));
    
}

int main() {
    srand(780538245);

    int screenWidth = 1600;
    int screenHeight = 800;

    InitWindow(screenWidth, screenHeight, "Handwritten Machine Learning");

    //SetTargetFPS(240);

    bool trainingMode = true;

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
    NeuralNet neuralNet = NeuralNet(5, 784, 300, 200, 100, 10);

    int expectedValue;

    expectedValue = dataset.getLabelBuffer()[0];
    loadGridFromDataset(grid, dataset, 0);
    loadGridValuesIntoNN(neuralNet, grid);
    neuralNet.run();

    float evaluatedCost;

    int currentImageIndex = 0;

    bool training = false;
    float learningRate = 0.001f;
    int batchSize = 50;
    int batches = 1200;
    int epochs = 50;
    
    int iterationsRan = 0;
    int batchesRan = 0;
    int epochsRan = 0;

    float previousEpochAccuracy = 0;
    int correctCount = 0;

    while (!WindowShouldClose()) {
        // Updates
        float delta = GetFrameTime();
        Vector2 mousePos = GetMousePosition();
        
        std::pair<int, int> cellCoords = grid.getCellCoords(mousePos);
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            if (cellCoords != std::pair<int, int>(-1, -1)) {
                //grid.setCellValue(cellCoords.first, cellCoords.second, 255);
                grid.paint(cellCoords.first, cellCoords.second);

                loadGridValuesIntoNN(neuralNet, grid);
                neuralNet.run();
            }
        }
        else if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
            if (cellCoords != std::pair<int, int>(-1, -1)) {
                //grid.setCellValue(cellCoords.first, cellCoords.second, 0);
                grid.erase(cellCoords.first, cellCoords.second);

                loadGridValuesIntoNN(neuralNet, grid);
                neuralNet.run();
            }
        }

        if (IsKeyPressed(KEY_RIGHT) && currentImageIndex < dataset.getImageCount() - 1) {
            currentImageIndex++;
            expectedValue = dataset.getLabelBuffer()[currentImageIndex];
            
            loadGridFromDataset(grid, dataset, currentImageIndex);
            loadGridValuesIntoNN(neuralNet, grid);
            neuralNet.run();
        }
        else if (IsKeyPressed(KEY_LEFT) && currentImageIndex > 0) {
            currentImageIndex--;
            expectedValue = dataset.getLabelBuffer()[currentImageIndex];
            
            loadGridFromDataset(grid, dataset, currentImageIndex);
            loadGridValuesIntoNN(neuralNet, grid);
            neuralNet.run();
        }

        if (IsKeyPressed(KEY_C)) {
            grid.clearGrid();
        }

        if (IsKeyPressed(KEY_SPACE)) {
            grid.invertBlackWhite();
        }

        // Training implementation for the purpose of being a demo
        if (IsKeyPressed(KEY_T) && !training) {
            training = true;
            iterationsRan = 0;
            batchesRan = 0;
            epochsRan = 0;
        }

        if (training) {
            // Show every individual iteration per frame
            currentImageIndex = batchesRan * batchSize + iterationsRan;
            expectedValue = dataset.getLabelBuffer()[currentImageIndex];

            loadGridFromDataset(grid, dataset, currentImageIndex);
            loadGridValuesIntoNN(neuralNet, grid);
            neuralNet.run();
            neuralNet.train(expectedValue);

            if (getHighestIndex(neuralNet.getOutputLayer()) == expectedValue) {
                correctCount++;
            }

            iterationsRan++;

            if (iterationsRan == batchSize) {
                neuralNet.applyGradients(learningRate, 1);

                iterationsRan = 0;

                batchesRan++;
                if (batchesRan == batches) {
                    batchesRan = 0;

                    previousEpochAccuracy = (float)correctCount / (batches * batchSize);
                    correctCount = 0;

                    epochsRan++;

                    if (epochsRan == epochs) {
                        training = false;

                        currentImageIndex = 0;
                        expectedValue = dataset.getLabelBuffer()[currentImageIndex];
                        loadGridFromDataset(grid, dataset, currentImageIndex);
                        loadGridValuesIntoNN(neuralNet, grid);
                        neuralNet.run();
                    }
                }
            }

            // Loop through one batch in one frame
            //for (int i = 0; i < batchSize; i++) {
            //    currentImageIndex = batchesRan * batchSize + i;
            //    expectedValue = dataset.getLabelBuffer()[currentImageIndex];

            //    loadGridFromDataset(grid, dataset, currentImageIndex);
            //    loadGridValuesIntoNN(neuralNet, grid);
            //    neuralNet.run();
            //    neuralNet.train(expectedValue);
            //}

            //neuralNet.applyGradients(learningRate, batchSize);

            //batchesRan++;
            //if (batchesRan == batches) {
            //    batchesRan = 0;

            //    epochsRan++;

            //    if (epochsRan == epochs) {
            //        training = false;

            //        currentImageIndex = 0;
            //        expectedValue = dataset.getLabelBuffer()[currentImageIndex];
            //        loadGridFromDataset(grid, dataset, currentImageIndex);
            //        loadGridValuesIntoNN(neuralNet, grid);
            //        neuralNet.run();
            //    }
            //}
        }
        
        // Testing loss function
        //JVector<float> maxError(10);
        //float testValues[10] = { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
        //for (int i = 0; i < 10; i++) {
        //    maxError[i] = testValues[i];
        //}
        //evaluatedCost = computeMeanError(expectedValue, maxError);

        //evaluatedCost = computeMeanError(expectedValue, neuralNet.getOutputLayer());
        evaluatedCost = computeCrossEntropy(expectedValue, neuralNet.getOutputLayer());
        
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

        std::string trainingData = "Training Info:\n\nEpochs ran: " + std::to_string(epochsRan) + "\n\nBatches ran: " + std::to_string(batchesRan) + "\n\nIterations ran: " + std::to_string(iterationsRan);
        DrawText(trainingData.c_str(), 1000, 200, 20, RED);

        JVector<std::pair<int, float>> confidenceValues(neuralNet.getOutputLayer().getSize());
        for (int i = 0; i < confidenceValues.getSize(); i++) {
            confidenceValues[i] = { i, neuralNet.getOutputLayer()[i] };
        }
        std::sort(confidenceValues.getDataPtr(), confidenceValues.getEnd(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) { return a.second > b.second; });

        DrawText("Confidence Values:", 800, 100, 20, BLUE);
        for (int i = 0; i < 10; i++) {
            float value = confidenceValues[i].second;//neuralNet.getOutputLayer()[i];

            std::string str = std::to_string(confidenceValues[i].first) + " " + std::to_string(value);

            DrawText(str.c_str(), 800, 140 + 30 * i, 20, BLUE);
        }

        std::string predictStr = "Predicted Number: " + std::to_string(confidenceValues[0].first);
        DrawText(predictStr.c_str(), 800, 500, 20, BLUE);

        float accuracy = (float)correctCount / (iterationsRan + batchesRan * batchSize);
        if ((iterationsRan + batchesRan * batchSize) == 0) {
            accuracy = 0;
        }
        std::string accuracyStr = "Previous Epoch Accuracy: " + std::to_string(previousEpochAccuracy) + "\nCurrent Epoch Accuracy: " + std::to_string(accuracy);
        DrawText(accuracyStr.c_str(), 800, 520, 20, GREEN);

        DrawFPS(10, 10);

        EndDrawing();
    }
}

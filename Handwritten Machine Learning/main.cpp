#include <string>
#include "raylib.h"
#include "rlgl.h"
#include <algorithm>
#include "MnistParser.h"
#include "PixelGrid.h"
#include "NeuralNet.h"
#include "NNMathLib.h"
#include "NNFileManager.h"

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
    grid.loadDatasetImage(dataset, 0);
    neuralNet.loadInGrid(grid);
    neuralNet.run();

    int currentImageIndex = 0;

    bool training = false;
    float learningRate = 0.001f;
    int batchSize = 10;
    int batches = 1;
    int epochs = 1000;
    
    int iterationsRan = 0;
    int batchesRan = 0;
    int epochsRan = 0;

    float previousEpochAccuracy = 0;
    int correctCount = 0;

    std::pair<int, int> prevCellCoords = { -1, -1 };

    while (!WindowShouldClose()) {
        // Updates
        float delta = GetFrameTime();
        Vector2 mousePos = GetMousePosition();
        
        std::pair<int, int> cellCoords = grid.getCellCoords(mousePos);
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            if (cellCoords != std::pair<int, int>(-1, -1) && cellCoords != prevCellCoords) {
                grid.paint(cellCoords.first, cellCoords.second, 40);

                neuralNet.loadInGrid(grid);
                neuralNet.run();

                prevCellCoords = cellCoords;
            }
        }
        else if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
            if (cellCoords != std::pair<int, int>(-1, -1) && cellCoords != prevCellCoords) {
                grid.erase(cellCoords.first, cellCoords.second, 30);

                neuralNet.loadInGrid(grid);
                neuralNet.run();

                prevCellCoords = cellCoords;
            }
        }

        if (IsKeyPressed(KEY_RIGHT) && currentImageIndex < dataset.getImageCount() - 1) {
            currentImageIndex++;
            expectedValue = dataset.getLabelBuffer()[currentImageIndex];
            
            grid.loadDatasetImage(dataset, currentImageIndex);
            neuralNet.loadInGrid(grid);
            neuralNet.run();
        }
        else if (IsKeyPressed(KEY_LEFT) && currentImageIndex > 0) {
            currentImageIndex--;
            expectedValue = dataset.getLabelBuffer()[currentImageIndex];
            
            grid.loadDatasetImage(dataset, currentImageIndex);
            neuralNet.loadInGrid(grid);
            neuralNet.run();
        }

        if (IsKeyPressed(KEY_C)) {
            grid.clearGrid();
        }

        if (IsKeyPressed(KEY_SPACE)) {
            grid.invertBlackWhite();
        }

        if (IsKeyPressed(KEY_S)) {
            saveNeuralNet(neuralNet);
        }

        if (IsKeyPressed(KEY_L)) {
            loadNeuralNet(neuralNet);
            neuralNet.loadInGrid(grid);
            neuralNet.run();
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

            grid.loadDatasetImage(dataset, currentImageIndex);
            neuralNet.loadInGrid(grid);
            neuralNet.run();
            neuralNet.train(expectedValue);

            if (neuralNet.getOutputLayer().getHighestIndex() == expectedValue) {
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
                        grid.loadDatasetImage(dataset, currentImageIndex);
                        neuralNet.loadInGrid(grid);
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

        JVector<float> expectedVector = makeHotVector(10, expectedValue);
        float evaluatedCost = computeCrossEntropy(expectedVector, neuralNet.getOutputLayer());
        
        // Sort confidence values from highest to lowest
        JVector<std::pair<int, float>> confidenceValues(neuralNet.getOutputLayer().getSize());
        for (int i = 0; i < confidenceValues.getSize(); i++) {
            confidenceValues[i] = { i, neuralNet.getOutputLayer()[i] };
        }
        std::sort(confidenceValues.getDataPtr(), confidenceValues.getEnd(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) { return a.second > b.second; });

        // Determine accuracy rate of current epoch
        float accuracy;
        if (iterationsRan + batchesRan == 0) {
            accuracy = 0;
        }
        else {
            accuracy = (float)correctCount / (iterationsRan + batchesRan * batchSize);
        }

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

        DrawText("Confidence Values:", 800, 100, 20, BLUE);
        for (int i = 0; i < 10; i++) {
            float value = confidenceValues[i].second;//neuralNet.getOutputLayer()[i];

            std::string str = std::to_string(confidenceValues[i].first) + " " + std::to_string(value);

            DrawText(str.c_str(), 800, 140 + 30 * i, 20, BLUE);
        }

        std::string predictStr = "Predicted Number: " + std::to_string(confidenceValues[0].first);
        DrawText(predictStr.c_str(), 800, 500, 20, BLUE);

        std::string accuracyStr = "Previous Epoch Accuracy: " + std::to_string(previousEpochAccuracy) + "\nCurrent Epoch Accuracy: " + std::to_string(accuracy);
        DrawText(accuracyStr.c_str(), 800, 520, 20, GREEN);

        DrawFPS(10, 10);

        EndDrawing();
    }
}

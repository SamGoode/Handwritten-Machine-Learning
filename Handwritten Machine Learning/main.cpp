#include <string>
#include "raylib.h"
#include "rlgl.h"
#include "PixelGrid.h"
#include "MnistParser.h"
#include <iostream>

int main() {
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

    int currentImageIndex = 0;
    bool updateGrid = true;

    while (!WindowShouldClose()) {
        // Updates
        float delta = GetFrameTime();
        Vector2 mousePos = GetMousePosition();
        
        std::pair<int, int> cellCoords = grid.getCellCoords(mousePos);
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            if (cellCoords != std::pair<int, int>(-1, -1)) {
                grid.setCellValue(cellCoords.first, cellCoords.second, 0);
            }
        }

        if (IsKeyPressed(KEY_RIGHT) && currentImageIndex < dataset.getImageCount() - 1) {
            currentImageIndex++;
            updateGrid = true;
        }
        else if (IsKeyPressed(KEY_LEFT) && currentImageIndex > 0) {
            currentImageIndex--;
            updateGrid = true;
        }

        if (updateGrid) {
            int offset = currentImageIndex * dataset.getRowCount() * dataset.getColumnCount();

            std::memcpy(grid.getDataPtr(), dataset.getImageBuffer() + offset, sizeof(byte) * dataset.getRowCount() * dataset.getColumnCount());

            updateGrid = false;
        }
        
        // Drawing
        BeginDrawing();

        ClearBackground(RAYWHITE);

        grid.draw();

        std::string cell = std::to_string(cellCoords.first) + ", " + std::to_string(cellCoords.second);
        DrawText(cell.c_str(), 10, 40, 20, GREEN);

        std::string imageIndexStr = "Image Index: " + std::to_string(currentImageIndex);
        DrawText(imageIndexStr.c_str(), 200, 20, 20, BLUE);

        std::string expectedValueStr = "Expected Value: " + std::to_string(dataset.getLabelBuffer()[currentImageIndex]);
        DrawText(expectedValueStr.c_str(), 200, 40, 20, BLUE);

        DrawFPS(10, 10);

        EndDrawing();
    }
}

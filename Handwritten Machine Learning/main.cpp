#include <string>
#include <fstream>
#include "raylib.h"
#include "rlgl.h"
#include "PixelGrid.h"
#include <iostream>

int reverseInt(int value) {
    using byte = unsigned char;

    byte a = value & 0x000000FF;
    byte b = (value >> 8) & 0x000000FF;
    byte c = (value >> 16) & 0x000000FF;
    byte d = (value >> 24) & 0x000000FF;

    return ((int)a << 24) + ((int)b << 16) + ((int)c << 8) + d;
}

using byte = unsigned char;
class mnistParser {
private:
    byte* imageBuffer;
    byte* labelBuffer;

    std::ifstream fileStream;

    int imageMagicNumber;
    int imageCount;
    int rowCount;
    int columnCount;

    int labelMagicNumber;
    int labelCount;

public:
    mnistParser() {
        imageBuffer = nullptr;
        labelBuffer = nullptr;
    }

    ~mnistParser() {
        delete[] imageBuffer;
        delete[] labelBuffer;
    }

    void loadImageBuffer(std::string fileName) {
        delete[] imageBuffer;

        fileStream.open(fileName, std::ios::binary);

        imageMagicNumber = 0;
        imageCount = 0;
        rowCount = 0;
        columnCount = 0;

        fileStream.read((char*)&imageMagicNumber, sizeof(imageMagicNumber));
        fileStream.read((char*)&imageCount, sizeof(imageCount));
        fileStream.read((char*)&rowCount, sizeof(rowCount));
        fileStream.read((char*)&columnCount, sizeof(columnCount));

        imageMagicNumber = reverseInt(imageMagicNumber);
        imageCount = reverseInt(imageCount);
        rowCount = reverseInt(rowCount);
        columnCount = reverseInt(columnCount);

        std::cout << imageMagicNumber << std::endl;
        std::cout << imageCount << std::endl;
        std::cout << rowCount << std::endl;
        std::cout << columnCount << std::endl;

        imageBuffer = new byte[imageCount * rowCount * columnCount];
        fileStream.read((char*)imageBuffer, sizeof(byte) * imageCount * rowCount * columnCount);

        fileStream.close();
    }

    void loadLabelBuffer(std::string fileName) {
        delete[] labelBuffer;

        fileStream.open(fileName, std::ios::binary);

        labelMagicNumber = 0;
        labelCount = 0;

        fileStream.read((char*)&labelMagicNumber, sizeof(labelMagicNumber));
        fileStream.read((char*)&labelCount, sizeof(labelCount));

        labelMagicNumber = reverseInt(labelMagicNumber);
        labelCount = reverseInt(labelCount);

        std::cout << labelMagicNumber << std::endl;
        std::cout << labelCount << std::endl;

        labelBuffer = new byte[labelCount];
        fileStream.read((char*)labelBuffer, sizeof(byte) * labelCount);

        fileStream.close();
    }

    byte* getImageBuffer() {
        return imageBuffer;
    }

    byte* getLabelBuffer() {
        return labelBuffer;
    }

    byte getImageCount() {
        return imageCount;
    }

    byte getRowCount() {
        return rowCount;
    }

    byte getColumnCount() {
        return columnCount;
    }

    byte getLabelCount() {
        return labelCount;
    }
};

int main() {
    int screenWidth = 1600;
    int screenHeight = 800;

    InitWindow(screenWidth, screenHeight, "Handwritten Machine Learning");

    SetTargetFPS(240);

    int seed = 844134593;
    //srand(seed);

    PixelGrid grid({ 100, 100 }, 600, 28);
    
    mnistParser dataset;

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

    dataset.loadImageBuffer(imagesFileName);
    dataset.loadLabelBuffer(labelsFileName);

    //std::ifstream file;
    //file.open(imagesFileName, std::ios::binary);

    //int imageCount = 0;
    //int rowCount = 0;
    //int columnCount = 0;

    //unsigned char* imageBuffer = nullptr;

    //if (file.is_open()) {
    //    int magicNumber = 0;
    //    file.read((char*)&magicNumber, sizeof(magicNumber));
    //    magicNumber = reverseInt(magicNumber);

    //    file.read((char*)&imageCount, sizeof(imageCount));
    //    imageCount = reverseInt(imageCount);

    //    file.read((char*)&rowCount, sizeof(rowCount));
    //    rowCount = reverseInt(rowCount);

    //    file.read((char*)&columnCount, sizeof(columnCount));
    //    columnCount = reverseInt(columnCount);

    //    std::cout << magicNumber << std::endl;
    //    std::cout << imageCount << std::endl;
    //    std::cout << rowCount << std::endl;
    //    std::cout << columnCount << std::endl;
    //    
    //    imageBuffer = new unsigned char[imageCount * rowCount * columnCount];

    //    file.read((char*)imageBuffer, sizeof(char) * imageCount * rowCount * columnCount);
    //}

    //file.close();

    //file.open(labelsFileName);

    //int labelCount = 0;

    //unsigned char* labelBuffer = nullptr;

    //if (file.is_open()) {
    //    int magicNumber = 0;
    //    file.read((char*)&magicNumber, sizeof(magicNumber));
    //    magicNumber = reverseInt(magicNumber);

    //    file.read((char*)&labelCount, sizeof(labelCount));
    //    imageCount = reverseInt(labelCount);

    //    std::cout << magicNumber << std::endl;
    //    std::cout << labelCount << std::endl;

    //    labelBuffer = new unsigned char[labelCount];

    //    file.read((char*)labelBuffer, sizeof(char) * labelCount);
    //}

    //file.close();


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
            for (int row = 0; row < dataset.getRowCount(); row++) {
                for (int col = 0; col < dataset.getColumnCount(); col++) {
                    int offset = currentImageIndex * dataset.getRowCount() * dataset.getColumnCount();

                    //unsigned char temp = imageBuffer[offset + col + row * columnCount];
                    unsigned char temp = dataset.getImageBuffer()[offset + col + row * dataset.getColumnCount()];

                    temp = 255 - temp;

                    grid.setCellValue(col, row, temp);
                }
            }

            updateGrid = false;
        }
        
        // Drawing
        BeginDrawing();

        ClearBackground(RAYWHITE);

        grid.draw();

        std::string cell = std::to_string(cellCoords.first) + ", " + std::to_string(cellCoords.second);
        DrawText(cell.c_str(), 10, 40, 20, GREEN);

        if (cellCoords == std::pair<int, int>(-1, -1)) {
            DrawText("false", 10, 60, 20, GREEN);
        }
        else {
            DrawText("true", 10, 60, 20, GREEN);
        }

        std::string imageIndexStr = "Image Index: " + std::to_string(currentImageIndex);
        DrawText(imageIndexStr.c_str(), 200, 20, 20, BLUE);

        std::string expectedValueStr = "Expected Value: " + std::to_string(dataset.getLabelBuffer()[currentImageIndex]);
        DrawText(expectedValueStr.c_str(), 200, 40, 20, BLUE);

        DrawFPS(10, 10);

        EndDrawing();
    }

    //// de allocate memory
    //delete[] imageBuffer;
    //delete[] labelBuffer;
}

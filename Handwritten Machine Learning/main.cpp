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

int main() {
    int screenWidth = 1600;
    int screenHeight = 800;

    InitWindow(screenWidth, screenHeight, "Smooth Particle Hydrodynamics Sim");

    //SetTargetFPS(240);

    int seed = 844134593;
    //srand(seed);

    //while (true) {
    //    PixelGrid grid({ 400, 100 }, 400, 10);
    //}

    PixelGrid grid({ 100, 100 }, 600, 28);
    //t10k-images.idx3-ubyte
    //std::ifstream file("datasets/train-images.idx3-ubyte", std::ios::binary);
    std::ifstream file("datasets/t10k-images.idx3-ubyte", std::ios::binary);

    if (file.is_open()) {
        int magicNumber = 0;
        file.read((char*)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);

        int imageCount = 0;
        file.read((char*)&imageCount, sizeof(imageCount));
        imageCount = reverseInt(imageCount);

        int rowCount = 0;
        file.read((char*)&rowCount, sizeof(rowCount));
        rowCount = reverseInt(rowCount);

        int columnCount = 0;
        file.read((char*)&columnCount, sizeof(columnCount));
        columnCount = reverseInt(columnCount);

        std::cout << magicNumber << std::endl;
        std::cout << imageCount << std::endl;
        std::cout << rowCount << std::endl;
        std::cout << columnCount << std::endl;

        for (int i = 0; i < 5000; i++) {
            for (int row = 0; row < rowCount; row++) {
                for (int col = 0; col < columnCount; col++) {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));

                    //temp = (unsigned char)(((float)col / columnCount) * 255);
                    //std::cout << (int)temp;

                    temp = 255 - temp;

                    grid.setCellValue(col, row, temp);
                }
                //std::cout << std::endl;
            }
        }
    }

    file.close();

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
            

        DrawFPS(10, 10);

        EndDrawing();
    }
}

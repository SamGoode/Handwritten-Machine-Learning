#include "raylib.h"
#include "rlgl.h"
#include "PixelGrid.h"
#include "string"

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

    PixelGrid grid({ 100, 100 }, 600, 16);

    while (!WindowShouldClose()) {
        // Updates
        float delta = GetFrameTime();

        Vector2 mousePos = GetMousePosition();
        
        std::pair<int, int> cellCoords = grid.getCellCoords(mousePos);
        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {

            if (cellCoords != std::pair<int, int>(-1, -1)) {
                grid.setCellValue(cellCoords.first, cellCoords.second, 1);
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

#pragma once
#include <math.h>
#include <utility>
#include "raylib.h"
#include "JMatrix.h"

class PixelGrid {
private:
	Vector2 pos;
	float size;
	int cellCount;

	float cellSize;

	JMatrix matrix;

public:
	PixelGrid(Vector2 _pos, float _size, int _cellCount);

	std::pair<int, int> getCellCoords(Vector2 screenPos);
	void setCellValue(int x, int y, char value);

	void draw();
};
#pragma once
#include <math.h>
#include <utility>
#include "raylib.h"
#include "JMatrix.h"

using byte = unsigned char;
class PixelGrid {
private:
	Vector2 pos;
	float size;
	int cellCount;
	bool shouldInvertBlackWhite;

	float cellSize;

	JMatrix<byte> matrix;

public:
	PixelGrid(Vector2 _pos, float _size, int _cellCount);

	JMatrix<byte>& getMatrix() { return matrix; }
	byte* getDataPtr() { return matrix.getDataPtr(); }

	std::pair<int, int> getCellCoords(Vector2 screenPos);
	void setCellValue(int x, int y, byte value);

	void invertBlackWhite();

	void draw();
};
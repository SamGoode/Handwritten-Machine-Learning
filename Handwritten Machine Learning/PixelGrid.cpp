#include "PixelGrid.h"

PixelGrid::PixelGrid(Vector2 _pos, float _size, int _cellCount) {
	pos = _pos;
	size = _size;
	cellCount = _cellCount;

	cellSize = size / cellCount;
	shouldInvertBlackWhite = false;

	matrix = JMatrix<byte>(cellCount, cellCount);
}

std::pair<int, int> PixelGrid::getCellCoords(Vector2 screenPos) {
	int x = floor((screenPos.x - pos.x) / cellSize);
	int y = floor((screenPos.y - pos.y) / cellSize);

	if (x < 0 || x >= cellCount || y < 0 || y >= cellCount) {
		x = -1;
		y = -1;
	}

	return std::pair<int, int>(x, y);
}

void PixelGrid::setCellValue(int x, int y, byte value) {
	matrix.setValue(x, y, value);
}

void PixelGrid::invertBlackWhite() {
	shouldInvertBlackWhite = !shouldInvertBlackWhite;
}

void PixelGrid::draw() {
	// Cell Colours
	for (int x = 0; x < cellCount; x++) {
		for (int y = 0; y < cellCount; y++) {
			byte value = matrix.getValue(x, y);

			if (shouldInvertBlackWhite) {
				value = 255 - value;
			}

			DrawRectangle(pos.x + x * cellSize, pos.y + y * cellSize, cellSize, cellSize, Color{value, value, value, 255});
		}
	}

	// Grid Lines
	for (int i = 1; i < cellCount; i++) {
		DrawLine(pos.x + i * cellSize, pos.y, pos.x + i * cellSize, pos.y + size, BLACK);
		DrawLine(pos.x, pos.y + i * cellSize, pos.x + size, pos.y + i * cellSize, BLACK);
	}

	// Outline
	DrawRectangleLines(pos.x, pos.y, size, size, BLACK);
}
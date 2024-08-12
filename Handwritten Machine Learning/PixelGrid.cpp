#include "PixelGrid.h"

PixelGrid::PixelGrid(Vector2 _pos, float _size, int _cellCount) {
	pos = _pos;
	size = _size;
	cellCount = _cellCount;

	cellSize = size / cellCount;
	shouldInvertBlackWhite = true;

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

void PixelGrid::paint(int x, int y, int strength) {
	for (int i = 0; i < 3; i++) {
		for (int n = 0; n < 3; n++) {
			if (x + i - 1 < 0 || x + i - 1 >= cellCount || y + n - 1 < 0 || y + n - 1 >= cellCount) {
				continue;
			}

			//matrix.addValue(x + i - 1, y + n - 1, gaussianKernel[n][i]);
			float currentValue = matrix.getValue(x + i - 1, y + n - 1);
			if (currentValue + gaussianKernel[n][i] * strength > 255) {
				matrix.setValue(x + i - 1, y + n - 1, 255);
			}
			else {
				matrix.setValue(x + i - 1, y + n - 1, currentValue + gaussianKernel[n][i] * strength);
			}
		}
	}
}

void PixelGrid::erase(int x, int y, int strength) {
	for (int i = 0; i < 3; i++) {
		for (int n = 0; n < 3; n++) {
			if (x + i - 1 < 0 || x + i - 1 >= cellCount || y + n - 1 < 0 || y + n - 1 >= cellCount) {
				continue;
			}

			//matrix.addValue(x + i - 1, y + n - 1, gaussianKernel[n][i]);
			float currentValue = matrix.getValue(x + i - 1, y + n - 1);
			if (currentValue - gaussianKernel[n][i] * strength < 0) {
				matrix.setValue(x + i - 1, y + n - 1, 0);
			}
			else {
				matrix.setValue(x + i - 1, y + n - 1, currentValue - gaussianKernel[n][i] * strength);
			}
		}
	}
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
#pragma once
#include <math.h>
#include <utility>
#include "raylib.h"

class JMatrix {
private:
	int width;
	int height;
	bool* values;

public:
	JMatrix() {}

	JMatrix(int _width, int _height) {
		width = _width;
		height = _height;

		values = new bool[width * height];
	}

	~JMatrix() {
		delete[] values;
	}

	JMatrix(const JMatrix& copy) {
		width = copy.width;
		height = copy.height;

		values = new bool[width * height];
		for (int i = 0; i < width * height; i++) {
			values[i] = copy.height;
		}
	}

	const JMatrix& operator=(const JMatrix& copy) {
		delete[] values;

		width = copy.width;
		height = copy.height;

		values = new bool[width * height];
		for (int i = 0; i < width * height; i++) {
			values[i] = copy.height;
		}

		return *this;
	}

	bool isValidCoord(int x, int y) {
		return !(x < 0 || x >= width || y < 0 || y >= height);
	}

	bool getValue(int x, int y) {
		if (!isValidCoord(x, y)) {
			throw "invalid cell coords";
		}

		return values[x + y * width];
	}

	void setValue(int x, int y, bool value) {
		if (!isValidCoord(x, y)) {
			throw "invalid cell coords";
		}

		values[x + y * width] = value;
	}

	void setAllValues(bool value) {
		for (int i = 0; i < width * height; i++) {
			values[i] = value;
		}
	}
};

class PixelGrid {
private:
	Vector2 pos;
	float size;
	int cellCount;

	float cellSize;

	JMatrix matrix;

public:
	PixelGrid(Vector2 _pos, float _size, int _cellCount);
	~PixelGrid();

	std::pair<int, int> getCellCoords(Vector2 screenPos);
	void setCellValue(int x, int y, bool value);

	void draw();
};
#include "JMatrix.h"

JMatrix::JMatrix(int _width, int _height) {
	width = _width;
	height = _height;

	values = new char[width * height];
}

JMatrix::JMatrix(const JMatrix& copy) {
	width = copy.width;
	height = copy.height;

	values = new char[width * height];
	for (int i = 0; i < width * height; i++) {
		values[i] = copy.height;
	}
}

const JMatrix& JMatrix::operator=(const JMatrix& copy) {
	delete[] values;

	width = copy.width;
	height = copy.height;

	values = new char[width * height];
	for (int i = 0; i < width * height; i++) {
		values[i] = copy.height;
	}

	return *this;
}

char JMatrix::getValue(int x, int y) {
	if (!isValidCoord(x, y)) {
		throw "invalid cell coords";
	}

	return values[x + y * width];
}

void JMatrix::setValue(int x, int y, char value) {
	if (!isValidCoord(x, y)) {
		throw "invalid cell coords";
	}

	values[x + y * width] = value;
}

void JMatrix::setAllValues(char value) {
	for (int i = 0; i < width * height; i++) {
		values[i] = value;
	}
}
#pragma once

class JMatrix {
private:
	int width;
	int height;
	char* values;

public:
	JMatrix() {}
	JMatrix(int _width, int _height);
	~JMatrix() { delete[] values; }
	JMatrix(const JMatrix& copy);

	const JMatrix& operator=(const JMatrix& copy);

	bool isValidCoord(int x, int y) { return !(x < 0 || x >= width || y < 0 || y >= height); }

	char getValue(int x, int y);
	void setValue(int x, int y, char value);
	void setAllValues(char value);
};
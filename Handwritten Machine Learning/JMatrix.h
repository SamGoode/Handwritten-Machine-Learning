#pragma once
#include <string>

template <typename T>
class JMatrix {
private:
	int columns;
	int rows;
	T* values;

public:
	JMatrix() {
		columns = 0;
		rows = 0;
		values = nullptr;
	}
	
	JMatrix(int _columns, int _rows) {
		columns = _columns;
		rows = _rows;
		values = new T[columns * rows];
	}

	~JMatrix() {
		delete[] values;
	}

	JMatrix(const JMatrix& copy) {
		columns = copy.columns;
		rows = copy.rows;
		values = new T[columns * rows];
		std::memcpy(values, copy.values, sizeof(T) * columns * rows);
		
	}

	const JMatrix& operator=(const JMatrix& copy) {
		delete[] values;

		columns = copy.columns;
		rows = copy.rows;
		values = new T[columns * rows];
		std::memcpy(values, copy.values, sizeof(T) * columns * rows);

		return *this;
	}

	T* getDataPtr() {
		return values;
	}

	int getColumnCount() {
		return columns;
	}

	int getRowCount() {
		return rows;
	}

	bool isValidCoord(int x, int y) {
		return !(x < 0 || x >= columns || y < 0 || y >= rows);
	}

	T getValue(int x, int y) {
		if (!isValidCoord(x, y)) {
			throw "out of bounds";
		}

		return values[x + y * columns];
	}

	void setValue(int x, int y, T value) {
		if (!isValidCoord(x, y)) {
			throw "out of bounds";
		}

		values[x + y * columns] = value;
	}

	void setAllValues(T value) {
		int totalCapacity = columns * rows;
		for (int i = 0; i < totalCapacity; i++) {
			values[i] = value;
		}
	}

	JMatrix multiply(JMatrix other) {
		if (columns != other.rows) {
			throw "invalid matrix dimensions";
		}
		int valuesCount = columns;
		JMatrix result(other.columns, rows);

		for (int row = 0; row < result.rows; row++) {
			for (int col = 0; col < result.columns; col++) {
				T value = 0;
				for (int i = 0; i < valuesCount; i++) {
					value += getValue(i, row) * other.getValue(col, i);
				}

				result.setValue(col, row, value);
			}
		}

		return result;
	}

	JMatrix add(JMatrix other) {
		if (columns != other.columns || rows != other.rows) {
			throw "invalid matrix dimensions";
		}

		JMatrix result(columns, rows);
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < columns; col++) {
				T value = getValue(col, row) + other.getValue(col, row);
				result.setValue(col, row, value);
			}
		}

		return result;
	}

	JMatrix transpose() {
		JMatrix result(rows, columns);

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < columns; col++) {
				T value = getValue(col, row);
				result.setValue(row, col, value);
			}
		}

		return result;
	}

	std::string toString() {
		std::string str;

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < columns; col++) {
				str += std::to_string(getValue(col, row)) + ",";
			}
			str += "\n";
		}

		return str;
	}
};
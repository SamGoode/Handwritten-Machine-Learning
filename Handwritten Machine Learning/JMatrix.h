#pragma once
#include <string>
#include "JVector.h"

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

	JMatrix(const JVector<T>& colVec, const JVector<T>& rowVec) {
		columns = colVec.getSize();
		rows = rowVec.getSize();
		values = new T[columns * rows];

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < columns; col++) {
				values[col + row * columns] = colVec[col] * rowVec[row];
			}
		}
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

	// For copying when matrices are of the same dimensions and memory reallocation is unnecessary.
	const JMatrix& copy(const JMatrix& copy) {
		if (columns != copy.columns || rows != copy.rows) {
			throw "invalid matrix dimensions";
		}

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

	JVector<T> multiply(JVector<T> vec) {
		if (columns != vec.getSize()) {
			throw "invalid dimensions";
		}
		
		JVector<T> result(rows);

		for (int row = 0; row < rows; row++) {
			T value = 0;
			for (int col = 0; col < columns; col++) {
				value += values[col + row * columns] * vec[col];
			}
			result.setValue(row, value);
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

	void addValue(int x, int y, T value) {
		if (!isValidCoord(x, y)) {
			throw "out of bounds";
		}

		values[x + y * columns] += value;
	}

	const JMatrix& addOn(JMatrix other) {
		if (columns != other.columns || rows != other.rows) {
			throw "invalid matrix dimensions";
		}

		int count = columns * rows;
		for (int i = 0; i < count; i++) {
			values[i] += other.values[i];
		}

		return *this;
	}

	const JMatrix& scale(float scalar) {
		int count = columns * rows;
		for (int i = 0; i < count; i++) {
			values[i] *= scalar;
		}

		return *this;
	}

	JMatrix transpose() {
		JMatrix result(rows, columns);

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < columns; col++) {
				result.setValue(row, col, getValue(col, row));
			}
		}

		return result;
	}

	// Multiply as if matrix was transposed
	JVector<T> transposedMultiply(JVector<T> vec) {
		if (rows != vec.getSize()) {
			throw "invalid dimensions";
		}

		JVector<T> result(columns);

		for (int col = 0; col < columns; col++) {
			T value = 0;
			for (int row = 0; row < rows; row++) {
				value += values[row + col * rows] * vec[row];
			}
			result.setValue(col, value);
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
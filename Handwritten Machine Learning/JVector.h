#pragma once

template <typename T>
class JVector {
private:
	int size;
	T* values;

public:
	JVector() {
		size = 0;
		values = nullptr;
	}

	JVector(int _size) {
		size = _size;
		values = new T[size];
	}

	~JVector() {
		delete[] values;
	}

	JVector(const JVector& copy) {
		size = copy.size;
		values = new T[size];
		std::memcpy(values, copy.values, sizeof(T) * size);
	}

	const JVector& operator=(const JVector& copy) {
		delete[] values;

		size = copy.size;
		values = new T[size];
		std::memcpy(values, copy.values, sizeof(T) * size);

		return *this;
	}

	// For copying when vectors are of the same length and memory reallocation is unnecessary.
	const JVector& copy(const JVector& copy) {
		if (size != copy.size) {
			throw "invalid vector size";
		}

		std::memcpy(values, copy.values, sizeof(T) * size);

		return *this;
	}

	T* getDataPtr() {
		return values;
	}

	T* getEnd() {
		return values + size;
	}

	int getSize() {
		return size;
	}

	const int getSize() const {
		return size;
	}

	T& operator[](int index) {
		if (index < 0 || index >= size) {
			throw "out of bounds";
		}

		return values[index];
	}

	const T& operator[](int index) const {
		if (index < 0 || index >= size) {
			throw "out of bounds";
		}

		return values[index];
	}

	T getValue(int index) {
		if (index < 0 || index >= size) {
			throw "out of bounds";
		}

		return values[index];
	}

	void setValue(int index, T value) {
		if (index < 0 || index >= size) {
			throw "out of bounds";
		}

		values[index] = value;
	}

	void setAllValues(T value) {
		for (int i = 0; i < size; i++) {
			values[i] = value;
		}
	}

	JVector add(const JVector& other) {
		if (size != other.size) {
			throw "invalid vector size";
		}

		JVector result(size);
		for (int i = 0; i < size; i++) {
			result[i] = values[i] + other.values[i];
		}

		return result;
	}

	const JVector& addValue(int index, T value) {
		if (index < 0 || index >= size) {
			throw "out of bounds";
		}

		values[index] += value;

		return *this;
	}

	const JVector& addOn(const JVector& other) const {
		if (size != other.size) {
			throw "invalid vector dimensions";
		}

		for (int i = 0; i < size; i++) {
			values[i] += other.values[i];
		}

		return *this;
	}

	const JVector& scale(float scalar) {
		for (int i = 0; i < size; i++) {
			values[i] *= scalar;
		}

		return *this;
	}

	int getHighestIndex() {
		int index = 0;
		float maxValue = 0;
		for (int i = 0; i < size; i++) {
			if (maxValue < values[i]) {
				index = i;
				maxValue = values[i];
			}
		}

		return index;
	}
};
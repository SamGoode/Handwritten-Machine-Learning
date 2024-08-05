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

	T* getDataPtr() {
		return values;
	}

	int getSize() {
		return size;
	}

	T operator[](int index) {
		return getValue(index);
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

	JVector add(JVector other) {
		if (size != other.size) {
			throw "invalid vector dimensions";
		}

		JVector result(size);
		for (int i = 0; i < size; i++) {
			T value = values[i] + other.values[i];
			result.setValue(i, value);
		}

		return result;
	}

	void addOn(JVector other) {
		if (size != other.size) {
			throw "invalid vector dimensions";
		}

		for (int i = 0; i < size; i++) {
			T value = values[i] + other.values[i];
			setValue(i, value);
		}
	}

	void scale(float scalar) {
		for (int i = 0; i < size; i++) {
			T value = values[i] * scalar;
			setValue(i, value);
		}
	}
};
#include "MnistParser.h"
#include <iostream>

int reverseInt(int value) {
    byte a = value & 0x000000FF;
    byte b = (value >> 8) & 0x000000FF;
    byte c = (value >> 16) & 0x000000FF;
    byte d = (value >> 24) & 0x000000FF;

    return ((int)a << 24) + ((int)b << 16) + ((int)c << 8) + d;
}

MnistParser::MnistParser() {
    imageBuffer = nullptr;
    labelBuffer = nullptr;
}

MnistParser::~MnistParser() {
    delete[] imageBuffer;
    delete[] labelBuffer;
}

void MnistParser::loadImageBuffer(std::string fileName) {
    delete[] imageBuffer;

    fileStream.open(fileName, std::ios::binary);

    imageMagicNumber = 0;
    imageCount = 0;
    rowCount = 0;
    columnCount = 0;

    fileStream.read((char*)&imageMagicNumber, sizeof(imageMagicNumber));
    fileStream.read((char*)&imageCount, sizeof(imageCount));
    fileStream.read((char*)&rowCount, sizeof(rowCount));
    fileStream.read((char*)&columnCount, sizeof(columnCount));

    imageMagicNumber = reverseInt(imageMagicNumber);
    imageCount = reverseInt(imageCount);
    rowCount = reverseInt(rowCount);
    columnCount = reverseInt(columnCount);

    std::cout << imageMagicNumber << std::endl;
    std::cout << imageCount << std::endl;
    std::cout << rowCount << std::endl;
    std::cout << columnCount << std::endl;

    imageBuffer = new byte[imageCount * rowCount * columnCount];
    fileStream.read((char*)imageBuffer, sizeof(byte) * imageCount * rowCount * columnCount);

    fileStream.close();
}

void MnistParser::loadLabelBuffer(std::string fileName) {
    delete[] labelBuffer;

    fileStream.open(fileName, std::ios::binary);

    labelMagicNumber = 0;
    labelCount = 0;

    fileStream.read((char*)&labelMagicNumber, sizeof(labelMagicNumber));
    fileStream.read((char*)&labelCount, sizeof(labelCount));

    labelMagicNumber = reverseInt(labelMagicNumber);
    labelCount = reverseInt(labelCount);

    std::cout << labelMagicNumber << std::endl;
    std::cout << labelCount << std::endl;

    labelBuffer = new byte[labelCount];
    fileStream.read((char*)labelBuffer, sizeof(byte) * labelCount);

    fileStream.close();
}
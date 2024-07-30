#pragma once
#include <fstream>

using byte = unsigned char;
class MnistParser {
private:
    byte* imageBuffer;
    byte* labelBuffer;

    std::ifstream fileStream;

    int imageMagicNumber;
    int imageCount;
    int rowCount;
    int columnCount;

    int labelMagicNumber;
    int labelCount;

public:
    MnistParser();
    ~MnistParser();

    void loadImageBuffer(std::string fileName);
    void loadLabelBuffer(std::string fileName);

    byte* getImageBuffer() { return imageBuffer; }
    byte* getLabelBuffer() { return labelBuffer; }

    int getImageCount() { return imageCount; }
    int getRowCount() { return rowCount; }
    int getColumnCount() { return columnCount; }
    int getLabelCount() { return labelCount; }
};
#pragma once
#include <fstream>
#include <vector>
#include <iostream>
#include <string>

// Define the Data Structure for an Image
struct MnistImage {
    std::vector<float> pixels; // 784 floats (0.0 to 1.0)
    int label;                 // 0-9
};

// Helper: Flip integer bytes (Big Endian -> Little Endian)
int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

std::vector<MnistImage> LoadMnistData(std::string imageFilename, std::string labelFilename) {
    std::vector<MnistImage> dataset;

    // Open streams
    std::ifstream file_images(imageFilename, std::ios::binary);
    std::ifstream file_labels(labelFilename, std::ios::binary);

    if (!file_images.is_open() || !file_labels.is_open()) {
        std::cout << "ERROR" << std::endl;
        return dataset;
    }

    // --- READ HEADERS ---
    int magicNum = 0, numImages = 0, rows = 0, cols = 0;
    int magicLabel = 0, numLabels = 0;

    // Images Header
    file_images.read((char*)&magicNum, 4);
    file_images.read((char*)&numImages, 4);
    file_images.read((char*)&rows, 4);
    file_images.read((char*)&cols, 4);

    // Labels Header
    file_labels.read((char*)&magicLabel, 4);
    file_labels.read((char*)&numLabels, 4);

    // Flip the bytes!
    magicNum = ReverseInt(magicNum);
    numImages = ReverseInt(numImages);
    rows = ReverseInt(rows);
    cols = ReverseInt(cols);
    numLabels = ReverseInt(numLabels);

    // --- READ DATA ---
    // Loop through all images
    for (int i = 0; i < numImages; ++i) {
        MnistImage img;

        // 1. Read Label (1 Byte)
        unsigned char label = 0;
        file_labels.read((char*)&label, 1);
        img.label = (int)label;

        // 2. Read Pixels (28*28 = 784 Bytes)
        for (int r = 0; r < (rows * cols); ++r) {
            unsigned char pixel = 0;
            file_images.read((char*)&pixel, 1);

            // Normalize: 0-255 -> 0.0-1.0
            img.pixels.push_back((float)pixel / 255.0f);
        }

        dataset.push_back(img);
    }

    return dataset;
}
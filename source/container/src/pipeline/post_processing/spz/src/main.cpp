#include "load-spz.h"
#include <iostream>
#include <filesystem>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::filesystem::path inputPath(inputFile);
    if (!std::filesystem::exists(inputPath)) {
        std::cerr << "Error: File does not exist: " << inputFile << std::endl;
        return 1;
    }

    std::string extension = inputPath.extension().string();
    std::filesystem::path outputPath;

    try {
        if (extension == ".ply") {
            // Convert PLY to SPZ
            spz::GaussianCloud cloud = spz::loadSplatFromPly(inputFile);
            outputPath = inputPath.replace_extension(".spz");
            if (!spz::saveSpz(cloud, outputPath.string())) {
                throw std::runtime_error("Failed to save SPZ file");
            }
        } else if (extension == ".spz") {
            // Convert SPZ to PLY
            spz::GaussianCloud cloud = spz::loadSpz(inputFile);
            outputPath = inputPath.replace_extension(".ply");
            if (!spz::saveSplatToPly(cloud, outputPath.string())) {
                throw std::runtime_error("Failed to save PLY file");
            }
        } else {
            std::cerr << "Error: Unsupported file extension: " << extension << std::endl;
            return 1;
        }

        std::cout << "Converted " << inputFile << " to " << outputPath << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

#include "load-spz.h"
#include <iostream>
#include <string>
#include <filesystem>
#include "export_macro.h"

extern "C" {
    SPZ_API bool ConvertSpzToPly(const char* spzFilePath, const char* plyFilePath) {
        try {
            std::string spzFile(spzFilePath);
            std::string plyFile(plyFilePath);
    
            if (!std::filesystem::exists(spzFile)) {
                std::cerr << "Error: SPZ file does not exist: " << spzFile << std::endl;
                return false;
            }
    
            spz::GaussianCloud cloud = spz::loadSpz(spzFile);
            if (!spz::saveSplatToPly(cloud, plyFile)) {
                std::cerr << "Error: Failed to save PLY file." << std::endl;
                return false;
            }
    
            return true;  // Success
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return false;
        }
    }
}

/*
// Expose the function for DLL
extern "C" __declspec(dllexport) bool ConvertSpzToPly(const char* spzFilePath, const char* plyFilePath) {
    try {
        std::string spzFile(spzFilePath);
        std::string plyFile(plyFilePath);

        if (!std::filesystem::exists(spzFile)) {
            std::cerr << "Error: SPZ file does not exist: " << spzFile << std::endl;
            return false;
        }

        spz::GaussianCloud cloud = spz::loadSpz(spzFile);
        if (!spz::saveSplatToPly(cloud, plyFile)) {
            std::cerr << "Error: Failed to save PLY file." << std::endl;
            return false;
        }

        return true;  // Success
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return false;
    }
}
*/
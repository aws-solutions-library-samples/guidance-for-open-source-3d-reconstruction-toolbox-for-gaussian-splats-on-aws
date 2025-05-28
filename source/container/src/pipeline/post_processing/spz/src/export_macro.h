#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #ifdef SPZ_CONVERTER_EXPORTS
        #define SPZ_API __declspec(dllexport)
    #else
        #define SPZ_API __declspec(dllimport)
    #endif
#else
    #define SPZ_API __attribute__((visibility("default")))
#endif

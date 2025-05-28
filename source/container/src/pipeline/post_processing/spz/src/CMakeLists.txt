cmake_minimum_required(VERSION 3.20)
project(SplatConverter)

set(CMAKE_CXX_STANDARD 20)

# Set visibility settings
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cc)

# Add source files
set(SOURCES
    cc/load-spz.cc
    cc/splat-c-types.cc
    cc/splat-types.cc
)

# Add executable
add_executable(splat_converter main.cpp ${SOURCES})

# Link zlib for compression/decompression
find_package(ZLIB REQUIRED)
target_link_libraries(splat_converter PRIVATE ZLIB::ZLIB)

# Shared library for Linux (.so)
add_library(spz_converter SHARED spz_converter_api.cpp ${SOURCES})
target_link_libraries(spz_converter ZLIB::ZLIB)

# Only use the DLL export definitions on Windows
if(WIN32)
    target_compile_definitions(spz_converter PRIVATE SPZ_CONVERTER_EXPORTS)
endif()

# Set installation paths
install(TARGETS splat_converter
    RUNTIME DESTINATION bin
)

install(TARGETS spz_converter
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install header files with correct paths
install(FILES 
    export_macro.h
    cc/load-spz.h
    cc/splat-c-types.h
    cc/splat-types.h
    DESTINATION include
)

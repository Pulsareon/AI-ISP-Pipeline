cmake_minimum_required(VERSION 3.10)
project(Pulsareon_AI_ISP)

set(CMAKE_CXX_STANDARD 14)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find ncnn
find_package(ncnn REQUIRED)

# OpenMP for multi-threading
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
endif()

add_executable(ai_isp ncnn_src/isp_ncnn.cpp)
target_link_libraries(ai_isp ncnn OpenCV::OpenCV)

# Optimization flags for ARM/Snapdragon
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    add_definitions(-march=armv8.2-a+fp16)
endif()

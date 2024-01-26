cmake_minimum_required(VERSION 3.28)

if (NOT CMAKE_GENERATOR MATCHES "Visual Studio")
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    endif()
endif()

project(avs-mlrt)

find_package(Boost 1.80 REQUIRED COMPONENTS filesystem)

set(Protobuf_USE_STATIC_LIBS ON)
find_package(Protobuf REQUIRED)
find_package(ONNX REQUIRED)

option(BUILD_MLRT_NCNN "Build mlrt_ncnn" ON)
message(STATUS "Build mlrt_ncnn: ${BUILD_MLRT_NCNN}.")

option(BUILD_MLRT_OV "Build mrt_ov" ON)
message(STATUS "Build mrt_ov: ${BUILD_MLRT_OV}.")

set(sources ${CMAKE_CURRENT_SOURCE_DIR}/common/onnx_utils.cpp)

if (BUILD_MLRT_NCNN)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mlrt_ncnn)
endif()

if (BUILD_MLRT_OV)
    set(sources
        ${sources}
        ${CMAKE_CURRENT_SOURCE_DIR}/common/convert_float_to_float16.cpp
    )
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mlrt_ov)
endif()

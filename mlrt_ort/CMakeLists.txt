cmake_minimum_required(VERSION 3.28)

project(mlrt_ort)

set(ONNX_RUNTIME_API_DIRECTORY "" CACHE PATH "Path to ONNX API headers")
set(ONNX_RUNTIME_LIB_DIRECTORY "" CACHE PATH "Path to ONNX Runtime libraries")
find_package(CUDAToolkit REQUIRED)

if (WIN32)
    set(sources
        ${sources}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mlrt_ort.rc
    )
endif()

add_library(mlrt_ort SHARED
    ${sources}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mlrt_ort.cpp
)

target_include_directories(mlrt_ort PRIVATE
    ${Boost_INCLUDE_DIRS}
    ${ONNX_INCLUDE_DIRS}
    ${ONNX_RUNTIME_API_DIRECTORY}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_directories(mlrt_ort PRIVATE ${ONNX_RUNTIME_LIB_DIRECTORY})

target_link_libraries(mlrt_ort PRIVATE
    Boost::filesystem
    onnx
    onnxruntime
    CUDA::cudart_static
    avisynth
)

if (MSVC)
    target_link_libraries(mlrt_ort PRIVATE delayimp)
    target_link_options(mlrt_ort PRIVATE "/DELAYLOAD:onnxruntime.dll")
endif()

target_compile_features(mlrt_ort PRIVATE cxx_std_20)

if (MSVC)
    target_compile_options(mlrt_ort PRIVATE /EHsc)
endif()

if (NOT CMAKE_GENERATOR MATCHES "Visual Studio")
    string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
    if (build_type STREQUAL debug)
        target_compile_definitions(mlrt_ort PRIVATE DEBUG_BUILD)
    else (build_type STREQUAL release)
        target_compile_definitions(mlrt_ort PRIVATE RELEASE_BUILD)
    endif()

    message(STATUS "Build type - ${CMAKE_BUILD_TYPE}")
endif()

if (WIN32)
    set_target_properties(mlrt_ort PROPERTIES
        PREFIX ""
        OUTPUT_NAME "mlrt_ort"
    )
else()
    find_package(Git)
    if (GIT_FOUND)
        execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
            OUTPUT_VARIABLE ver
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set_target_properties(mlrt_ort PROPERTIES OUTPUT_NAME "mlrt_ort.${ver}")
    endif()

    include(GNUInstallDirs)
    INSTALL(TARGETS mlrt_ort LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth")

    # uninstall target
    if(NOT TARGET uninstall)
    configure_file(
        "${PROJECT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
    endif()
endif()

cmake_minimum_required(VERSION 3.28)

project(mlrt_ncnn)

find_package(ncnn REQUIRED)

if (WIN32)
    set(sources
        ${sources}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mlrt_ncnn.rc
    )
endif()

add_library(mlrt_ncnn SHARED
    ${sources}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mlrt_ncnn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx2ncnn.cpp
)

target_include_directories(mlrt_ncnn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Boost_INCLUDE_DIRS}
    ${ONNX_INCLUDE_DIRS}
)

target_link_libraries(mlrt_ncnn PRIVATE
    Boost::filesystem
    onnx
    ncnn
    avisynth
)

target_compile_features(mlrt_ncnn PRIVATE cxx_std_17)

if (MSVC)
    target_compile_options(mlrt_ncnn PRIVATE /EHsc)
endif()

if (NOT CMAKE_GENERATOR MATCHES "Visual Studio")
    string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
    if (build_type STREQUAL debug)
        target_compile_definitions(mlrt_ncnn PRIVATE DEBUG_BUILD)
    else (build_type STREQUAL release)
        target_compile_definitions(mlrt_ncnn PRIVATE RELEASE_BUILD)
    endif()

    message(STATUS "Build type - ${CMAKE_BUILD_TYPE}")
endif()

if (WIN32)
    set_target_properties(mlrt_ncnn PROPERTIES
        PREFIX ""
        OUTPUT_NAME "mlrt_ncnn"
    )
else()
    find_package(Git)
    if (GIT_FOUND)
        execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
            OUTPUT_VARIABLE ver
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set_target_properties(mlrt_ncnn PROPERTIES OUTPUT_NAME "mlrt_ncnn.${ver}")
    endif()

    include(GNUInstallDirs)
    INSTALL(TARGETS mlrt_ncnn LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth")

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

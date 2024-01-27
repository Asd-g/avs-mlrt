cmake_minimum_required(VERSION 3.28)

project(mlrt_ov)

find_package(OpenVINO REQUIRED REQUIRED)

if (WIN32)
    set(sources
        ${sources}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mlrt_ov.rc
    )
endif()

add_library(mlrt_ov SHARED
    ${sources}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mlrt_ov.cpp
)

target_include_directories(mlrt_ov PRIVATE
    ${Boost_INCLUDE_DIRS}
    ${ONNX_INCLUDE_DIRS}
)

target_link_libraries(mlrt_ov PRIVATE
    Boost::filesystem
    onnx
    openvino::runtime
    avisynth
)

target_compile_features(mlrt_ov PRIVATE cxx_std_17)

if (MSVC)
    target_compile_options(mlrt_ov PRIVATE /EHsc)
endif()

if (NOT CMAKE_GENERATOR MATCHES "Visual Studio")
    string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
    if (build_type STREQUAL debug)
        target_compile_definitions(mlrt_ov PRIVATE DEBUG_BUILD)
    else (build_type STREQUAL release)
        target_compile_definitions(mlrt_ov PRIVATE RELEASE_BUILD)
    endif()

    message(STATUS "Build type - ${CMAKE_BUILD_TYPE}")
endif()

if (WIN32)
    set_target_properties(mlrt_ov PROPERTIES
        PREFIX ""
        OUTPUT_NAME "mlrt_ov"
    )
else()
    find_package(Git)
    if (GIT_FOUND)
        execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
            OUTPUT_VARIABLE ver
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set_target_properties(mlrt_ov PROPERTIES OUTPUT_NAME "mlrt_ov.${ver}")
    endif()

    include(GNUInstallDirs)
    INSTALL(TARGETS mlrt_ov LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth")

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

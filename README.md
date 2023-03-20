## Description

This project provides AviSynth+ ML filter runtimes for variety of platforms.

This is [a partial port of the VapourSynth plugin vs-mlrt](https://github.com/AmusementClub/vs-mlrt).

To simplify usage, a wrapper [mlrt.avsi](https://github.com/Asd-g/avs-mlrt/blob/main/mlrt.avsi) is provided for all bundled models.

Custom models can be found [in this doom9 thread](https://forum.doom9.org/showthread.php?t=184768).

### Requirements:

- Vulkan device (mlrt_ncnn only)

- Intel GPU (mlrt_ov only, `device="GPU"` only)

- AviSynth+ r3928 (can be downloaded from [here](https://gitlab.com/uvz/AviSynthPlus-Builds) until official release is uploaded) or later

- Microsoft VisualC++ Redistributable Package 2022 (can be downloaded from [here](https://github.com/abbodi1406/vcredist/releases))

### mlrt_ncnn [usage](https://github.com/Asd-g/avs-mlrt/blob/main/README_ncnn.md)

### mlrt_ov [usage](https://github.com/Asd-g/avs-mlrt/blob/main/README_ov.md)

### Building:

    Requirements:
        - Vulkan SDK (https://vulkan.lunarg.com/sdk/home#windows) (mlrt_ncnn only)
        - ncnn (mlrt_ncnn only)
        - OpenVINO (https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) (mlrt_ov only)
        - protobuf
        - onnx
        - boost

- Windows
    ```
    Build steps:
        Install Vulkan SDk. (mlrt_ncnn only)
        Download the latest ncnn release. (mlrt_ncnn only)
        Download the latest OpenVINO release. (mlrt_ov only)

        Building protobuf:
            cmake -B build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release -D protobuf_BUILD_SHARED_LIBS=OFF  -D protobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -DCMAKE_INSTALL_PREFIX=../install
            cmake --build build_rel
            cmake --install build_rel

        Building onnx:
            cmake -B build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release -DONNX_USE_LITE_PROTO=ON -DONNX_USE_PROTOBUF_SHARED_LIBS=OFF -DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_ML=0 -DCMAKE_INSTALL_PREFIX=../install -DONNX_NAMESPACE=ONNX_NAMESPACE -DCMAKE_PREFIX_PATH=../install
            cmake --build build_rel
            cmake --install build_rel

        Building boost:
            b2 --with-system --with-filesystem --with-chrono -q --toolset=msvc-14.3 address-model=64 variant=release link=static runtime-link=shared threading=multi --hash --prefix=.\bin\x64
            b2 --with-system --with-filesystem --with-chrono -q --toolset=msvc-14.3 address-model=64 variant=release link=static runtime-link=shared threading=multi --hash --prefix=.\bin\x64 install

        Use solution files.
    ```

- Linux
    ```

    ```

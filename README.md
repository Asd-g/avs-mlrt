## Description

This project provides AviSynth+ ML filter runtimes for variety of platforms.

This is [a partial port of the VapourSynth plugin vs-mlrt](https://github.com/AmusementClub/vs-mlrt).

To simplify usage, a wrapper [mlrt.avsi](https://github.com/Asd-g/avs-mlrt/blob/main/mlrt.avsi) is provided for all bundled models.

Custom models can be found [in this doom9 thread](https://forum.doom9.org/showthread.php?t=184768).

### Requirements:

- Vulkan device

- AviSynth+ r3682 (can be downloaded from [here](https://gitlab.com/uvz/AviSynthPlus-Builds) until official release is uploaded) (r3689 recommended) or later

- Microsoft VisualC++ Redistributable Package 2022 (can be downloaded from [here](https://github.com/abbodi1406/vcredist/releases))

### Usage:

```
mlrt_ncnn(clip[] input, string "network_path", int "overlap_w", int "overlap_h", int "tilesize_w", int "tilesize_h", int "device_id", int "num_streams", bool "builtin", string "builtindir", bool "fp16", bool "path_is_serialization", bool "list_gpu")
```

### Parameters:

- input\
    Clips to process.\
    They must be in RGB/Gray 32-bit planar format, have same dimensions and same number of frames.

- network_path\
    Path to the model.

- overlap_w, overlap_h\
    Overlap width and overlap height of the tiles, respectively.\
    Must be less than or equal to `tilesize_w` / `tilesize_h` `/` `2`.\
    Default: 0.

- tilesize_w, tilesize_h\
    Tile width and height, respectively.\
    Use smaller value to reduce GPU memory usage.\
    Must be specified when `overlap_w` / `overlap_h` > 0.\
    Default: input_width, input_height.

- device_id\
    GPU device to use.\
    By default the default device is selected.

- num_streams\
    GPU parallel execution.\
    Default: 1.

- builtin\
    Whether the models are in the same location with the plugin.\
    Default: True.

- builtindir\
    Root folder when `builtin` is used.\
    Default: "models".

- fp16\
    Enable FP16 mode.\
    Default: False.

- path_is_serialization\
    Whether the model is serialized into one contiguous memory buffer.\
    Default: False.

- list_gpu\
    Simply print a list of available GPU devices on the frame and does nothing else.\
    Default: False.

### Building:

    Requirements:
        - Vulkan SDK (https://vulkan.lunarg.com/sdk/home#windows)
        - protobuf
        - onnx
        - ncnn
        - boost

- Windows
    ```
    Build steps:
        Install Vulkan SDk.

        Building protobuf:
            cmake -B build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release -D protobuf_BUILD_SHARED_LIBS=OFF  -D protobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -DCMAKE_INSTALL_PREFIX=../install
            cmake --build build_rel
            cmake --install build_rel

        Building onnx:
            cmake -B build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release -DONNX_USE_LITE_PROTO=ON -DONNX_USE_PROTOBUF_SHARED_LIBS=OFF -DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_ML=0 -DCMAKE_INSTALL_PREFIX=../install -DONNX_NAMESPACE=ONNX_NAMESPACE -DCMAKE_PREFIX_PATH=../install
            cmake --build build_rel
            cmake --install build_rel

        Download the latest ncnn release.

        Building boost:
            b2 --with-system --with-filesystem --with-chrono -q --toolset=msvc-14.3 address-model=64 variant=release link=static runtime-link=shared threading=multi --hash --prefix=.\bin\x64
            b2 --with-system --with-filesystem --with-chrono -q --toolset=msvc-14.3 address-model=64 variant=release link=static runtime-link=shared threading=multi --hash --prefix=.\bin\x64 install

        Use solution files to build avs_mlrt_ncnn.
    ```

- Linux
    ```

    ```

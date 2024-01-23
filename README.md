## Description

This project provides AviSynth+ ML filter runtimes for variety of platforms.

This is [a partial port of the VapourSynth plugin vs-mlrt](https://github.com/AmusementClub/vs-mlrt).

To simplify usage, a wrapper [mlrt.avsi](https://github.com/Asd-g/avs-mlrt/blob/main/mlrt.avsi) is provided for all bundled models.

Custom models can be found [in this doom9 thread](https://forum.doom9.org/showthread.php?t=184768).

### Requirements:

- Vulkan device (mlrt_ncnn only)

- Intel GPU (mlrt_ov only, `device="GPU"` only)

- AviSynth+ r3928 or later ([1](https://github.com/AviSynth/AviSynthPlus/releases) / [2](https://forum.doom9.org/showthread.php?t=181351) / [3](https://gitlab.com/uvz/AviSynthPlus-Builds))

- Microsoft VisualC++ Redistributable Package 2022 (can be downloaded from [here](https://github.com/abbodi1406/vcredist/releases))

### Filters:

#### mlrt_ncnn

[ncnn](https://github.com/Tencent/ncnn) is a popular AI inference runtime. mlrt_ncnn provides a vulkan based runtime for some AI filters.

It includes support for on-the-fly ONNX to ncnn native format conversion so as to provide a unified interface across all runtimes provided by this project.

[How to use mlrt_ncnn](https://github.com/Asd-g/avs-mlrt/blob/main/mlrt_ncnn/README_ncnn.md).

#### mlrt_ov

[OpenVINO](https://docs.openvino.ai/latest/index.html) is an AI inference runtime developed by Intel, mainly targeting x86 CPUs and Intel GPUs.

The mlrt_ov plugin provides optimized pure CPU & Intel GPU runtime for some popular AI filters. Intel GPU supports Gen 8+ on Broadwell+ and the Arc series GPUs.

[How to use mlrt_ov](https://github.com/Asd-g/avs-mlrt/blob/main/mlrt_ov/README_ov.md).

### Building:

Requirements:
- Vulkan SDK (https://vulkan.lunarg.com/sdk/home#windows) (mlrt_ncnn only)
- ncnn (mlrt_ncnn only)
- OpenVINO (https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) (mlrt_ov only)
- protobuf
- onnx
- boost

```
# Build steps:
# Install Vulkan SDK. (mlrt_ncnn only)
# Get the latest ncnn release. (mlrt_ncnn only)
# Get the latest OpenVINO release. (mlrt_ov only)

git clone --recurse-submodules https://github.com/Asd-g/avs-mlrt
cd avs-mlrt

# Building protobuf:
cd protobuf
cmake -B build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release -D protobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -DCMAKE_INSTALL_PREFIX=../install
cmake --build build_rel
cmake --install build_rel

# Building onnx:
cd ..\onnx
cmake -B build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release -DONNX_USE_LITE_PROTO=ON -DONNX_USE_PROTOBUF_SHARED_LIBS=OFF -DONNX_GEN_PB_TYPE_STUBS=OFF -DONNX_ML=0 -DCMAKE_INSTALL_PREFIX=../install -DONNX_NAMESPACE=ONNX_NAMESPACE -DCMAKE_PREFIX_PATH=../install
cmake --build build_rel
cmake --install build_rel

# Building boost (optional):
#b2 --with-system --with-filesystem --with-chrono -q --toolset=msvc-14.3 address-model=64 variant=release link=static runtime-link=shared threading=multi --hash --prefix=.\bin\x64
#b2 --with-system --with-filesystem --with-chrono -q --toolset=msvc-14.3 address-model=64 variant=release link=static runtime-link=shared threading=multi --hash --prefix=.\bin\x64 install

# Building avs-mlrt
# BUILD_MRLT_NCNN=ON (default)
# BUILD_MLRT_OV=ON (default)
cd ..
cmake -B build -G Ninja -DProtobuf_DIR=<current_location>\install\cmake -DONNX_DIR=<current_location>\install\lib\cmake\ONNX -Dncnn_DIR=c:\uc\avs-mlrt\install\lib\cmake\ncnn -DOpenVINO_DIR=c:\uc\w_openvino_toolkit_windows_2022.3.0.9052.9752fafe8eb_x86_64\runtime\cmake
```

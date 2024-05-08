## Description

This project provides AviSynth+ ML filter runtimes for variety of platforms.

This is [a partial port of the VapourSynth plugin vs-mlrt](https://github.com/AmusementClub/vs-mlrt).

To simplify usage, [wrappers](https://github.com/Asd-g/avs-mlrt/tree/main/scripts) are provided for all bundled models.

Custom models can be found [in this doom9 thread](https://forum.doom9.org/showthread.php?t=184768).

### Requirements:

- Vulkan device (`mlrt_ncnn`)

- Intel GPU (`mlrt_ov(... device="GPU")`)

- NVIDIA GPU (`mlrt_ort(... provider="cuda")`)

- DirectX 12-compatible hardware (`mlr_ort(... provider="dml")`)

- AviSynth+ r3928 or later ([1](https://github.com/AviSynth/AviSynthPlus/releases) / [2](https://forum.doom9.org/showthread.php?t=181351) / [3](https://gitlab.com/uvz/AviSynthPlus-Builds))

- Microsoft VisualC++ Redistributable Package 2022 (can be downloaded from [here](https://github.com/abbodi1406/vcredist/releases))

### Filters:

#### mlrt_ncnn

[ncnn](https://github.com/Tencent/ncnn) is a popular AI inference runtime. mlrt_ncnn provides a vulkan based runtime for some AI filters.

It includes support for on-the-fly ONNX to ncnn native format conversion so as to provide a unified interface across all runtimes provided by this project.

[How to use mlrt_ncnn](https://github.com/Asd-g/avs-mlrt/blob/main/mlrt_ncnn/README.md).

#### mlrt_ov

[OpenVINO](https://docs.openvino.ai/latest/index.html) is an AI inference runtime developed by Intel, mainly targeting x86 CPUs and Intel GPUs.

The mlrt_ov plugin provides optimized pure CPU & Intel GPU runtime for some popular AI filters. Intel GPU supports Gen 8+ on Broadwell+ and the Arc series GPUs.

[How to use mlrt_ov](https://github.com/Asd-g/avs-mlrt/blob/main/mlrt_ov/README.md).

#### mlrt_ort

[ONNX Runtime](https://onnxruntime.ai/) is an AI inference runtime with many backends.

The mlrt_ort plugin provides optimized CPU, CUDA GPU and DX12 GPU runtime for some popular AI filters.

[How to use mlrt_ort](https://github.com/Asd-g/avs-mlrt/blob/main/mlrt_ort/README.md).

### Benchmarks

Device specific benchmarks can be found [here](https://github.com/AmusementClub/vs-mlrt/wiki#device-specific-benchmarks).

DirectML benchmarks can be found [here](https://github.com/AmusementClub/vs-mlrt/releases/tag/v13.2).

### Building:

Requirements:
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) (`mlrt_ncnn`)
- [ncnn](https://github.com/Tencent/ncnn) (mlrt_ncnn only)
- [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) (`mlrt_ov`)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) (`mlrt_ort`)
- [protobuf](https://github.com/protocolbuffers/protobuf)
- [onnx](https://github.com/onnx/onnx)
- [boost](https://www.boost.org/)

```
# Build steps:
# Get the latest Vulkan SDK. (mlrt_ncnn)
# Get the latest ncnn release. (mlrt_ncnn)
# Get the latest OpenVINO release. (mlrt_ov)
# Get the latest Microsoft.ML.OnnxRuntime.DirectML. (mlrt_ort)
# Get the latest CUDA Toolkit. (mlrt_ort)

git clone --recurse-submodules --depth 1 https://github.com/Asd-g/avs-mlrt
cd avs-mlrt

# Building protobuf:
cd protobuf
cmake -B build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -DCMAKE_INSTALL_PREFIX=../install
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

# Building onnxruntime (mlrt_ort)
cd ..\mlrt_ort\onnxruntime
cmake -B build_rel -DCMAKE_BUILD_TYPE=Release -G Ninja -Donnxruntime_BUILD_SHARED_LIB=ON -Dprotobuf_BUILD_TESTS=OFF -Donnxruntime_BUILD_UNIT_TESTS=OFF -Donnxruntime_USE_CUDA=ON -Donnxruntime_CUDA_HOME="%CUDA_PATH%" -Donnxruntime_CUDNN_HOME=<...> -DCMAKE_INSTALL_PREFIX=..\..\install -DCMAKE_CUDA_ARCHITECTURES="50;61-real;75-real;86-real;89-real" -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -DONNX_USE_PROTOBUF_SHARED_LIBS=OFF cmake -Donnxruntime_USE_DML=ON -DFLATBUFFERS_BUILD_TESTS=OFF cmake
cmake --build build_rel
cmake --install build_rel

# Building avs-mlrt
# BUILD_MRLT_NCNN=ON (default)
# BUILD_MLRT_OV=ON (default)
# BUILD_MLRT_ORT=ON (default)
cd ..\..
cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=".\install";<Vulkan_SDK_path>;<boost_path> -Dncnn_DIR=<...> -DOpenVINO_DIR=<...>
```

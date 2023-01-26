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

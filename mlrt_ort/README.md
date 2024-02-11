## mlrt_ort

### Runtime files

1. Download [LoadDLL.dll](https://forum.doom9.org/showthread.php?t=173259).
2. Download the ONNX runtime files (`onnxruntime_dll.7z`) from [Releases](https://github.com/Asd-g/avs-mlrt/releases).
3. Extract them in `ort_runtime_files` folder (the folder can be located next to `mlrt_ort.dll` or in other place).
4. (optional if `provider="cuda"` will be used) Download the CUDA runtime files (`cuda_dll.7z`) from [Releases](https://github.com/Asd-g/avs-mlrt/releases) and extract them in the `ort_runtime_files` folder.
5. (optional if `provide="dml"` will NOT be used) Delete `DirectML.dll` from the `ort_runtime_files` folder.
6. Create a `mlrt_ort_loader.avsi` script next to `mlrt_ort.dll` that contains (the order of dll loading is important):

```
# Uncomment the following lines if provider="cuda" will be used.
#LoadDLL("path_to_ort_runtime_files\cublasLt64_12.dll")
#LoadDLL("path_to_ort_runtime_files\cudart64_12.dll")
#LoadDLL("path_to_ort_runtime_files\cudnn64_8.dll")
#LoadDLL("path_to_ort_runtime_files\cufft64_11.dll")
#LoadDLL("path_to_ort_runtime_files\cudnn_ops_infer64_8.dll")
#LoadDLL("path_to_ort_runtime_files\cudnn_cnn_infer64_8.dll")
#LoadDLL("path_to_ort_runtime_files\cudnn_adv_infer64_8.dll")

# Uncomment the following line if provider="dml" will be used.
#LoadDLL("path_to_ort_runtime_files\DirectML.dll")

LoadDLL("path_to_ort_runtime_files\onnxruntime.dll")
LoadPlugin("mlrt_ort.dll")
```

### Usage:

```
mlrt_ort(clip[] input, string "network_path", int "overlap_w", int "overlap_h", int "tilesize_w", int "tilesize_h", string "provider", int "device", int "num_streams", int "verbosity", bool "cudnn_benchmark", bool "builtin", string "builtindir", bool "fp16", bool "path_is_serialization", bool "use_cuda_graph", string[] "fp16_blacklist_ops")
```

### Parameters:

- input<br>
    Clips to process.<br>
    They must be in RGB/Gray 32-bit planar format, have same dimensions and same number of frames.

- network_path<br>
    Path to the model.

- overlap_w, overlap_h<br>
    Overlap width and overlap height of the tiles, respectively.<br>
    Must be less than or equal to `tilesize_w` / `tilesize_h` `/` `2`.<br>
    Default: 0.

- tilesize_w, tilesize_h<br>
    Tile width and height, respectively.<br>
    Use smaller value to reduce GPU memory usage.<br>
    Must be specified when `overlap_w` / `overlap_h` > 0.<br>
    Default: input_width, input_height.

- provider<br>
    Specifies the device to run the inference on.<br>
    `"CPU"` or `""`: pure CPU backend.<br>
    `"CUDA"`: CUDA GPU backend.<br>
    `"DML"`: DirectX 12 GPU backed.<br>
    Default: `""`.

- device<br>
    Select the GPU device for the CUDA backend.<br>
    Default: 0.

- num_streams<br>
    GPU parallel execution.<br>
    Must be greater than 0.<br>
    Default: 1.

- verbosity<br>
    Specify the verbosity of logging, the default is warning.<br>
    0: Everything, ORT_LOGGING_LEVEL_VERBOSE.<br>
    1: Info, ORT_LOGGING_LEVEL_INFO.<br>
    2: Warnings, ORT_LOGGING_LEVEL_WARNING.<br>
    3: Errors, ORT_LOGGING_LEVEL_ERROR.<br>
    4: Fatal error only, ORT_LOGGING_LEVEL_FATAL.<br>
    Default: 3.

- cudnn_benchmark<br>
    Whether to let cuDNN use benchmarking to search for the best convolution kernel to use.<br>
    It might incur some startup latency.<br>
    Default: True.

- builtin<br>
    Whether the models are in the same location with the plugin.<br>
    Default: True.

- builtindir<br>
    Root folder when `builtin` is used.<br>
    Default: "models".

- fp16<br>
    Enable FP16 mode.<br>
    Default: False.

- path_is_serialization<br>
    Whether the model is serialized into one contiguous memory buffer.<br>
    Default: False.

- use_cuda_graph<br>
    Whether to use CUDA Graphs to improve performance and reduce CPU overhead in CUDA backend.<br>
    Not all models are supported.<br>
    Default: False.

- fp16_blacklist_ops<br>
    Configurable FP16 operations black list.<br>
    Default: `["ArrayFeatureExtractor", "Binarizer", "CastMap", "CategoryMapper", "DictVectorizer", "FeatureVectorizer", "Imputer", "LabelEncoder", "LinearClassifier", "LinearRegressor", "Normalizer", "OneHotEncoder", "SVMClassifier", "SVMRegressor", "Scaler", "TreeEnsembleClassifier", "TreeEnsembleRegressor", "ZipMap", "NonMaxSuppression", "TopK", "RoiAlign", "Range", "CumSum", "Min", "Max", "Resize", "Upsample", "ReduceMean", "GridSample"]`.

When `overlap` and `tilesize` are not specified, the filter will internally try to resize the network to fit the input clips.<br>
This might not always work (for example, the network might require the width to be divisible by 8), and the filter will error out in this case.

The general rule is to either:
1. Left out `overlap`, `tilesize` at all and just process the input frame in one tile, or
2. set them so that the frame is processed in `tilesize_w` x `tilesize_h` tiles, and adjacent tiles will have an overlap of `overlap_w` x `overlap_h` pixels on each direction. The overlapped region will be throw out so that only internal output pixels are used.

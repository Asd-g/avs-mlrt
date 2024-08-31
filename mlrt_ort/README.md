## mlrt_ort

### Runtime files

All runtime files must be in `mlrt_ort_rt` folder that is located in the same folder as `mlrt_ort.dll`.<br>
They can be downloaded from [Releases](https://github.com/Asd-g/avs-mlrt/releases).

- For `mlrt_ort(provider="cpu")` - only `onnxruntime_dll.7z` is needed.
- For `mlrt_ort(provider="dml")` - `onnxruntime_dll.7z` and `dml_dll.7z` are needed.
- For `mlrt_ort(provider="cuda")` - `onnxruntime_dll.7z` and `cuda_dll.7z` are needed.
- For all providers (`mlrt_ort(provider="cpu/dml/cuda")`) - `onnxruntime_dll.7z`, `dml_dll.7z` and `cuda_dll.7z` are needed.

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

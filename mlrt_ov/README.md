## mlrt_ov

### Runtime files

All runtime files must be in `mlrt_ov_rt` folder that is located in the same folder as `mlrt_ov.dll`.<br>
They can be downloaded from [Releases](https://github.com/Asd-g/avs-mlrt/releases) (` openvino_dll.7z`).

### Note:

The plugin isn't tested with `deviec="GPU"` at all due to lack of hardware.

### Usage:

```
mlrt_ov(clip[] input, string "network_path", int "overlap_w", int "overlap_h", int "tilesize_w", int "tilesize_h", string "device", bool "builtin", string "builtindir", bool "fp16", string "config", bool "path_is_serialization", bool "list_devices", string[] "fp16_blacklist_ops", string "dot_path")
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

- device<br>
    Device to use - CPU or GPU.<br>
    For example, if there are more than one GPU device, to use the first device - `"GPU.0"`, to use the second device - `"GPU.1"`<br>
    Default: "CPU".

- builtin<br>
    Whether the models are in the same location with the plugin.<br>
    Default: True.

- builtindir<br>
    Root folder when `builtin` is used.<br>
    Default: "models".

- fp16<br>
    Enable FP16 mode.<br>
    Default: False.

- config<br>
    Configuration parameters.<br>
    CPU configuration parameters can be found [here](https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_supported_plugins_CPU.html#supported-configuration-parameters).<br>
    GPU configuration parameters can be found [here](https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_supported_plugins_GPU.html#supported-configuration-parameters).<br>
    `KEY_` prefix must be omitted.<br>
    Format is: `param=value`.<br>
    If more than one parameter is specified, the parameters must be separated by space.

    For example, to disable all internal CPU threading: `config="CPU_THROUGHPUT_STREAMS=0 CPU_THREADS_NUM=1 CPU_BIND_THREAD=NO"`

- path_is_serialization<br>
    Whether the model is serialized into one contiguous memory buffer.<br>
    Default: False.

- list_devices<br>
    Simply print a list of available CPU/GPU devices on the frame and does nothing else.<br>
    Default: False.

- fp16_blacklist_ops<br>
    Configurable FP16 operations black list.<br>
    Default: ["ArrayFeatureExtractor", "Binarizer", "CastMap", "CategoryMapper", "DictVectorizer", "FeatureVectorizer", "Imputer", "LabelEncoder", "LinearClassifier", "LinearRegressor", "Normalizer", "OneHotEncoder", "SVMClassifier", "TreeEnsembleRegressor", "ZipMap", "NonMaxSuppression", "TopK", "RoiAlign", "Range", "CumSum", "Min", "Max"].

- dot_path<br>
    Path for .dot file.<br>
    Allows to serialize to xDot format.

When `overlap` and `tilesize` are not specified, the filter will internally try to resize the network to fit the input clips.<br>
This might not always work (for example, the network might require the width to be divisible by 8), and the filter will error out in this case.

The general rule is to either:
1. Left out `overlap`, `tilesize` at all and just process the input frame in one tile, or
2. set them so that the frame is processed in `tilesize_w` x `tilesize_h` tiles, and adjacent tiles will have an overlap of `overlap_w` x `overlap_h` pixels on each direction. The overlapped region will be throw out so that only internal output pixels are used.

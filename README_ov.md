## mlrt_ov

Download the required OpenVINO runtimes from [here](https://github.com/Asd-g/avs-mlrt/blob/main/2022.3.7z).

After there are few options:
- Add the extracted files to PATH.
- Place the extracted files in the same location as `mlrt_ov.dll`.
- (Requires [LoadDLL.dll](https://forum.doom9.org/showthread.php?t=173259)) Create `AutoLoadDll.avsi` with following:
```
LoadDLL("path_to\tbb.dll")
LoadDLL("path_to\openvino.dll")
LoadPlugin("mlrt_ov.dll")
```
Then you can place `AutoLoadDll.avsi` in the plugins folder for autoloading or you can import it manually.

### Note:

The plugin isn't tested with `deviec="GPU"` at all due to lack of hardware.

### Usage:

```
mlrt_ov(clip[] input, string "network_path", int "overlap_w", int "overlap_h", int "tilesize_w", int "tilesize_h", string "device", bool "builtin", string "builtindir", bool "fp16", string "config", bool "path_is_serialization", bool "list_devices", string[] "fp16_blacklist_ops", string "dot_path")
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

- device\
    Device to use - CPU or GPU.\
    Default: "CPU".

- builtin\
    Whether the models are in the same location with the plugin.\
    Default: True.

- builtindir\
    Root folder when `builtin` is used.\
    Default: "models".

- fp16\
    Enable FP16 mode.\
    Default: False.

- config\
    Configuration parameters.\
    CPU configuration parameters can be found [here](https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_supported_plugins_CPU.html#supported-configuration-parameters).\
    GPU configuration parameters can be found [here](https://docs.openvino.ai/2021.4/openvino_docs_IE_DG_supported_plugins_GPU.html#supported-configuration-parameters).\
    `KEY_` prefix must be omitted.\
    Format is: `param=value`.\
    If more than one parameter is specified, the parameters must be separated by space.

    For example, to disable all internal CPU threading: `config="CPU_THROUGHPUT_STREAMS=0 CPU_THREADS_NUM=1 CPU_BIND_THREAD=NO"`

- path_is_serialization\
    Whether the model is serialized into one contiguous memory buffer.\
    Default: False.

- list_devices\
    Simply print a list of available CPU/GPU devices on the frame and does nothing else.\
    Default: False.

- fp16_blacklist_ops\
    Configurable FP16 operations black list.\
    Default: ["ArrayFeatureExtractor", "Binarizer", "CastMap", "CategoryMapper", "DictVectorizer", "FeatureVectorizer", "Imputer", "LabelEncoder", "LinearClassifier", "LinearRegressor", "Normalizer", "OneHotEncoder", "SVMClassifier", "TreeEnsembleRegressor", "ZipMap", "NonMaxSuppression", "TopK", "RoiAlign", "Range", "CumSum", "Min", "Max"].

- dot_path\
    Path for .dot file.\
    Allows to serialize to xDot format.

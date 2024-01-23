## mlrt_ncnn

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

## mlrt_ncnn

### Usage:

```
mlrt_ncnn(clip[] input, string "network_path", int "overlap_w", int "overlap_h", int "tilesize_w", int "tilesize_h", int "device_id", int "num_streams", bool "builtin", string "builtindir", bool "fp16", bool "path_is_serialization", bool "list_gpu")
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

- device_id<br>
    GPU device to use.<br>
    By default the default device is selected.

- num_streams<br>
    GPU parallel execution.<br>
    Default: 1.

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

- list_gpu<br>
    Simply print a list of available GPU devices on the frame and does nothing else.<br>
    Default: False.

When `overlap` and `tilesize` are not specified, the filter will internally try to resize the network to fit the input clips.<br>
This might not always work (for example, the network might require the width to be divisible by 8), and the filter will error out in this case.

The general rule is to either:
1. Left out `overlap`, `tilesize` at all and just process the input frame in one tile, or
2. set them so that the frame is processed in `tilesize_w` x `tilesize_h` tiles, and adjacent tiles will have an overlap of `overlap_w` x `overlap_h` pixels on each direction. The overlapped region will be throw out so that only internal output pixels are used.

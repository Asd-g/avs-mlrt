### A avs-mlrt wrapper of SAFA.
### This is a port of the https://github.com/AmusementClub/vs-mlrt/blob/master/scripts/vsmlrt.py


### Requirements
# avs_mlrt
# mlrt_common.avsi


### Usage ###
###
# mlrt_SAFA(clip c, val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", int "adaptive")
###
## Parameters ##
#---------------
# c: Input clip.
# Must be in RGB 32-bit planar format.
#---------------
# tiles, tilesize (default not specified): Tiles size.
# If specified as single int, the values is used for both tile width and tile height.
# If specified as array, the first element is referred to tile width.
# If tilesize is specified, tiles doesn't have effect.
#---------------
# overlap_w, overlap_h (default overlap_w: 16): Overlap width and overlap height of the tiles, respectively.
# overlap_h default: overlap_w.
#---------------
# model (default 0): What model to be used.
# Folder "models" must be in the same location as mlrt_xxx.dll.
/*
0: v0.1
1: v0.2
2: v0.3
3: v0.4
*/
#---------------
# backend (default "ov"): What backend to be used (ov, or ort) and its tuning parameters.
# It can be specified as single string.
# If specified as array of strings, the first element must be the backend type and the rest one are the tuning parameters.
# ncnn is not supported.
# For example, backend=["ov",  "fp16=true"] - backend type is "ov" and fp16 mode is True.
#---------------
# adaptive (mlrt_SAFA) (default 0).
# 0: non_adaptive
# 1: adaptive1x - use adaptive path only at 1x scale
# 2: adaptive - use adaptive path at 1x, 1/2x and 1/4x scales, proposed algorithm


### Version: 1.1.6


### Changelog ###
#---------------
# 1.1.6
# Moved every filter in own file.
#---------------
# 1.1.5
# mlrt_SAFA: allowed backend="ort".
#---------------
# 1.1.4
# Added support for mlrt_ort.
#---------------
# 1.1.3
# mlrt_W2x: added more swin_unet_x models.
# mlrt_RealESRGAN: added esrgan janai models.
# Added mlrt_SAFA.
#---------------
# Added support for padding any video dimension to mod4. (mlrt_W2x(model=6, scale=2))
#---------------
# Clamped the input to [0,1] before processing.
# mlrt_W2x(model=6, scale=2): padded the input (4, 4, 4, 4) before processing.
#---------------
# mlrt_W2x: added swin_unet_art models.
# Added backend "ov".
# Removed parameter backend_args.
#---------------
# mlrt_W2x: fixed the model path when scale=1.
#---------------
# Initial version.


Function mlrt_SAFA(clip c, val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", int "adaptive")
{
    model = Default(model, 0)
    overlap_w = Default(overlap_w, 16)
    overlap_h = Default(overlap_h, overlap_w)
    backend = backend_define(Default(backend, "ov"), "mlrt_SAFA")
    adaptive = Default(adaptive, 0)

    Assert(ComponentSize(c) == 4, "mlrt_SAFA: clip must be in 32-bit planar format.")
    Assert(IsPlanarRGB(c), "mlrt_SAFA: clip must be of planar RGB color family.")
    Assert(model >= 0 && model <= 3, "mlrt_SAFA: model must be between 0..3.")
    Assert(backend[0] == "ov" || backend[0] == "ort", "mlrt_SAFA: backend must be ov or ort.")
    Assert(adaptive >= 0 || adaptive <= 2, "mlrt_SAFA: adaptive must be between 0..2.")

    safa_models = Select(model,
        \ "v0.1",
        \ "v0.2",
        \ "v0.3",
        \ "v0.4")

    adaptive_models = Select(adaptive,
        \ "non_adaptive",
        \ "adaptive1x",
        \ "adaptive")

    multiple = (model <= 2) ? 8 : 16

    tile_overlap = calc_tilesize(Width(c), Height(c), multiple, overlap_w, overlap_h, tiles, tilesize)

    Assert(!(tile_overlap[0] % multiple != 0 || tile_overlap[1] % multiple != 0), "mlrt_SAFA: tile size must be divisible by " + String(multiple) + "(" + String(tile_overlap[0]) + ", " + String(tile_overlap[1]) + ").")

    network_path = "safa/" + "safa_" + safa_models + "_" + adaptive_models + ".onnx"

    backend_defaults = backend_defaults(backend)

    c = Expr(c, "x 0 1 clip")

    source_frame_num = FrameCount(c)
    half_frame_num = BitRShiftL(source_frame_num - 1, 1)
    clips = (source_frame_num % 2 == 0) ? [Trim(c, 0, half_frame_num), Trim(c, half_frame_num + 1, 0)] : [Trim(c, 0, half_frame_num), Trim(c, half_frame_num + 1, 0) + Trim(c, source_frame_num, -1)]

    clip2x = Eval("mlrt_" + backend[0] + "(clips, network_path=network_path, overlap_w=tile_overlap[2], overlap_h=tile_overlap[3], tilesize_w=tile_overlap[0], tilesize_h=tile_overlap[1], builtin=true" + backend_defaults + ")")

    c = Interleave(Crop(clip2x, 0, 0, 0, BitRShiftL(Height(clip2x), 1)), Crop(clip2x, 0, BitRShiftL(Height(clip2x), 1), 0, 0))

    if (source_frame_num != FrameCount(c))
    {
        c = Trim(c, 0, FrameCount(c) - 2)
    }

    return c
}

### A avs-mlrt wrapper of DPIR.
### This is a port of the https://github.com/AmusementClub/vs-mlrt/blob/master/scripts/vsmlrt.py


### Requirements
# avs_mlrt
# mlrt_common.avsi


### Usage ###
###
# mlrt_DPIR(clip c, val "strength", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend")
###
## Parameters ##
#---------------
# c: Input clip.
# Must be in RGB 32-bit planar format or in Gray 32-bit format (model=0/2).
#---------------
# strength (default 5.0): Strength of denoising.
# It may be specified as clip (histogram mask).
#---------------
# tiles, tilesize (default not specified): Tiles size.
# If specified as single int, the values is used for both tile width and tile height.
# If specified as array, the first element is referred to tile width.
# If tilesize is specified, tiles doesn't have effect.
#---------------
# overlap_w, overlap_h (overlap_w default: 0): Overlap width and overlap height of the tiles, respectively.
# overlap_h default: overlap_w.
#---------------
# model (default 0): What model to be used.
# Folder "models" must be in the same location as mlrt_xxx.dll.
/*
0: "drunet_gray"
1: "drunet_color"
2: "drunet_deblocking_grayscale"
3: "drunet_deblocking_color"
*/
#---------------
# backend (default "ncnn"): What backend to be used (ncnn, ov, or ort) and its tuning parameters.
# It can be specified as single string.
# If specified as array of strings, the first element must be the backend type and the rest one are the tuning parameters.
# For example, backend=["ncnn",  "fp16=true"] - backend type is "ncnn" and fp16 mode is True.


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


Function mlrt_DPIR(clip c, val "strength", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend")
{
    model = Default(model, 0)
    overlap_w = Default(overlap_w, 0)
    overlap_h = Default(overlap_h, overlap_w)
    backend = backend_define(backend, "mlrt_DPIR")

    Assert(ComponentSize(c) == 4, "mlrt_DPIR: clip must be in 32-bit planar format.")
    Assert(model >= 0 && model <= 3, "mlrt_DPIR: model must be 0, 1, 2, or 3.")

    dpir_models = Select(model, "drunet_gray", "drunet_color", "drunet_deblocking_grayscale", "drunet_deblocking_color")

    Assert(!((model == 0 || model == 2) && NumComponents(c) > 1), "mlrt_DPIR: clip must be of GRAY color family.")
    Assert(!((model == 1 || model == 3) && !IsPlanarRGB(c)), "mlrt_DPIR: clip must be of RGB color family.")

    if (!Defined(strength))
    {
        strength = 5.0
    }
    if (IsClip(strength))
    {
        Assert(NumComponents(strength) == 1, "mlrt_DPIR: strength must be of GRAY color family.")
        Assert(!(Width(strength) != Width(c) || Height(strength) != Height(c)), "mlrt_DPIR: strength must be of the same size as clip.")
        Assert(FrameCount(strength) == FrameCount(c), "mlrt_DPIR: strength must be of the same length as clip.")

        strength = Expr(strength, "x 255 /")
    }
    else
    {
        strength = strength
        Assert(IsFloat(strength) || IsInt(strength), "mlrt_DPIR: strength must be a float or a clip.")

        strength = strength / 255.0
        strength = KillAudio(BlankClip(c, pixel_type="y32", colors=[strength, strength, strength]))
        strength = Loop(Trim(strength, 0, -1), FrameCount(c))
    }

    multiple = 8

    tile_overlap = calc_tilesize(Width(c), Height(c), multiple, overlap_w, overlap_h, tiles, tilesize)

    Assert(!(tile_overlap[0] % multiple != 0 || tile_overlap[1] % multiple != 0), "mlrt_DPIR: tile size must be divisible by " + String(multiple) + "(" + String(tile_overlap[0]) + ", " + String(tile_overlap[1]) + ").")

    network_path = "dpir/" + dpir_models + ".onnx"
    backend_defaults = backend_defaults(backend)

    c = Expr(c, "x 0 1 clip")
    return Eval("mlrt_" + backend[0] + "([c, strength], network_path=network_path, overlap_w=tile_overlap[2], overlap_h=tile_overlap[3], tilesize_w=tile_overlap[0], tilesize_h=tile_overlap[1], builtin=true" + backend_defaults + ")")
}

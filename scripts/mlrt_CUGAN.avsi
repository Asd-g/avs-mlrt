### A avs-mlrt wrapper of CUGAN.
### This is a port of the https://github.com/AmusementClub/vs-mlrt/blob/master/scripts/vsmlrt.py


### Requirements
# avs_mlrt
# mlrt_common.avsi


### Usage ###
###
# mlrt_CUGAN(clip c, int "noise", int "scale", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", val "backend", int "version", bool "conformance")
###
## Parameters ##
#---------------
# c: Input clip.
# Must be in RGB 32-bit planar format.
#---------------
# noise (default -1): Denoise level.
# Large value means strong denoise effect, -1 - no effect.
# Must be between -1..3.
#---------------
# scale (default 2): Upscale ratio.
# Must be 2, 3, or 4.
#---------------
# tiles, tilesize (default not specified): Tiles size.
# If specified as single int, the values is used for both tile width and tile height.
# If specified as array, the first element is referred to tile width.
# If tilesize is specified, tiles doesn't have effect.
#---------------
# overlap_w, overlap_h (default 4): Overlap width and overlap height of the tiles, respectively.
# overlap_h default: overlap_w.
#---------------
# backend (default "ncnn"): What backend to be used (ncnn, ov, or ort) and its tuning parameters.
# It can be specified as single string.
# If specified as array of strings, the first element must be the backend type and the rest one are the tuning parameters.
# For example, backend=["ncnn",  "fp16=true"] - backend type is "ncnn" and fp16 mode is True.
#---------------
# version (default 1).
# 1: legacy
# 2: pro (only models for "noise" in [-1, 0, 3] and "scale" in [2, 3] are published currently)
#---------------
# conformance (default True): Currently specifies dynamic range compression for cugan-pro.


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


Function mlrt_CUGAN(clip c, int "noise", int "scale", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", val "backend", int "version", bool "conformance")
{
    /*
    denoising strength: 0 < -1 < 1 < 2 < 3
    version: (1 or 2)
        1 -> legacy,
        2 -> pro (only models for "noise" in [-1, 0, 3] and "scale" in [2, 3] are published currently)
    */

    noise = Default(noise, -1)
    scale = Default(scale, 2)
    overlap_w = Default(overlap_w, 4)
    overlap_h = Default(overlap_h, overlap_w)
    version = Default(version, 1)
    conformance = Default(conformance, True) # currently specifies dynamic range compression for cugan-pro
    backend = backend_define(backend, "mlrt_CUGAN")

    Assert(ComponentSize(c) == 4, "mlrt_CUGAN: clip must be in 32-bit planar format.")
    Assert(IsPlanarRGB(c), "mlrt_CUGAN: clip must be of planar RGB color family.")
    Assert(noise >= -1 && noise <= 3, "mlrt_CUGAN: noise must be -1, 0, 1, 2, or 3.")
    Assert(scale == 2 || scale == 3 || scale == 4, "mlrt_CUGAN: scale must be 2, 3, or 4.")
    Assert(version == 1 || version == 2, "mlrt_CUGAN: version must be 1 (legacy) or 2 (pro)")
    Assert(!(scale != 2 && (noise == 1 || noise == 2)), "mlrt_CUGAN: scale=" + String(scale) + " model.\nmlrt_CUGAN does not support noise reduction level " + String(noise) + ".")

    multiple = 2

    tile_overlap = calc_tilesize(Width(c), Height(c), multiple, overlap_w, overlap_h, tiles, tilesize)

    Assert(!(tile_overlap[0] % multiple != 0 || tile_overlap[1] % multiple != 0), "mlrt_CUGAN: tile size must be divisible by " + String(multiple) + "(" + String(tile_overlap[0]) + ", " + String(tile_overlap[1]) + ").")

    folder_path = "cugan/"

    model_name = (version == 1) ? (noise == -1) ? ("up" + String(scale) + "x-latest-no-denoise.onnx") : \
    (noise == 0) ? ("up" + String(scale) + "x-latest-conservative.onnx") : \
    ("up" + String(scale) + "x-latest-denoise" + String(noise) + "x.onnx") : \
    (noise == -1) ? ("pro-no-denoise3x-up" + String(scale) + "x.onnx") : \
    (noise == 0) ? ("pro-conservative-up" + String(scale) + "x.onnx") : \
    ("pro-denoise" + String(noise) + "x-up" + String(scale) + "x.onnx")

    # https://github.com/bilibili/ailab/blob/e102bef22384c629f82552dbec3d6b5bab125639/Real-CUGAN/upcunet_v3.py#L1275-L1276
    if (conformance && version == 2)
    {
        c = Expr(c, "x 0.7 * 0.15 +")
    }

    backend_defaults = backend_defaults(backend)

    c = Expr(c, "x 0 1 clip")
    c = Eval("mlrt_" + backend[0] + "(c, network_path=folder_path+model_name, overlap_w=tile_overlap[2], overlap_h=tile_overlap[3], tilesize_w=tile_overlap[0], tilesize_h=tile_overlap[1], builtin=true" + backend_defaults + ")")

    # https://github.com/bilibili/ailab/blob/e102bef22384c629f82552dbec3d6b5bab125639/Real-CUGAN/upcunet_v3.py#L269
    if (conformance && version == 2)
    {
        c = Expr(c, "x 0.15 - 0.7 /")
    }

    return c
}

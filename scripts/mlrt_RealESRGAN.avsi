### A avs-mlrt wrapper of RealESRGAN/RealESRGANv2.
### This is a port of the https://github.com/AmusementClub/vs-mlrt/blob/master/scripts/vsmlrt.py


### Requirements
# avs_mlrt
# mlrt_common.avsi
# fmtconv - for mlrt_RealESRGAN(scale=...)


### Usage ###
###
# mlrt_RealESRGAN/mlrt_RealESRGANv2(clip c, val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", float "scale")
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
# overlap_w, overlap_h (overlap_w default: 8): Overlap width and overlap height of the tiles, respectively.
# overlap_h default: overlap_w.
#---------------
# model (default 0): What model to be used.
# Folder "models" must be in the same location as mlrt_xxx.dll.
/*
0: "animevideo-xsx2" # v2
1: "animevideo-xsx4" # v2
2: "animevideov3" # v3 4x
# contributed: janaiV2(2x) https://github.com/the-database/mpv-upscale-2x_animejanai/releases/tag/2.0.0 maintainer: hooke007
3: "animejanaiV2L1"
4: "animejanaiV2L2"
5: "animejanaiV2L3"
# contributed: janaiV3-hd(2x) https://github.com/the-database/mpv-upscale-2x_animejanai/releases/tag/3.0.0 maintainer: hooke007
6: "animejanaiV3-HD-L1"
7: "animejanaiV3-HD-L2"
8: "animejanaiV3-HD-L3"
*/
#---------------
# backend (default "ncnn"): What backend to be used (ncnn, ov, or ort) and its tuning parameters.
# It can be specified as single string.
# If specified as array of strings, the first element must be the backend type and the rest one are the tuning parameters.
# For example, backend=["ncnn",  "fp16=true"] - backend type is "ncnn" and fp16 mode is True.
#---------------
# scale (default not specified): Upscale ratio.


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


Function mlrt_RealESRGAN(clip c, val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", float "scale")
{
    model = Default(model, 0)
    overlap_w = Default(overlap_w, 8)
    overlap_h = Default(overlap_h, overlap_w)
    backend = backend_define(backend, "mlrt_RealESRGAN")

    Assert(ComponentSize(c) == 4, "mlrt_RealESRGAN: clip must be in 32-bit planar format.")
    Assert(IsPlanarRGB(c), "mlrt_RealESRGAN: clip must be of planar RGB color family.")
    Assert(model >= 0 && model <= 8, "mlrt_RealESRGAN: model must be between 0..8.")

    # v2, v2, v3 4x
    realesrgan_models = Select(model,
        \ "RealESRGANv2-animevideo-xsx2",
        \ "RealESRGANv2-animevideo-xsx4",
        \ "realesr-animevideov3",
        \ "animejanaiV2L1",
        \ "animejanaiV2L2",
        \ "animejanaiV2L3",
        \ "animejanaiV3-HD-L1",
        \ "animejanaiV3-HD-L2",
        \ "animejanaiV3-HD-L3")

    multiple = 1

    tile_overlap = calc_tilesize(Width(c), Height(c), multiple, overlap_w, overlap_h, tiles, tilesize)

    network_path = "RealESRGANv2/" + realesrgan_models + ".onnx"

    backend_defaults = backend_defaults(backend)

    width_ = Width(c)
    height_ = height(c)
    c = Expr(c, "x 0 1 clip")
    c = Eval("mlrt_" + backend[0] + "(c, network_path=network_path, overlap_w=tile_overlap[2], overlap_h=tile_overlap[3], tilesize_w=tile_overlap[0], tilesize_h=tile_overlap[1], builtin=true" + backend_defaults + ")")

    if (Defined(scale))
    {
        scale_h = Width(c) / width_
        scale_v = Height(c) / height_

        Assert(scale_h == scale_v, "mlrt_RealESRGAN: scale_h must be equal to scale_v.")

        if (scale != scale_h)
        {
            rescale = scale / scale_h

            c = (rescale > 1) ? fmtc_resample(c, scale=rescale, kernel="lanczos", taps=4) : \
            fmtc_resample(c, scale=rescale, kernel="lanczos", taps=4, fh=1/rescale, fv=1/rescale)
        }
    }

    return c
}

Function mlrt_RealESRGANv2(clip c, val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", float "scale")
{
    return mlrt_RealESRGAN(c, tiles, tilesize, overlap_w, overlap_h, model, backend, scale)
}

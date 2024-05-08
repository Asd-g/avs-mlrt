### A avs-mlrt wrapper of waifu2x.
### This is a port of the https://github.com/AmusementClub/vs-mlrt/blob/master/scripts/vsmlrt.py


### Requirements
# avs_mlrt
# mlrt_common.avsi
# avsresize - mlrt_W2x(model=0/1/2, preprocess=true), mlrt_W2x(model=8, scale < 4)
# fmtconv - for mlrt_W2x(model=6, scale=1)
# FillBorders - only for mlrt_W2x(model=6, scale=2)


### Usage ###
###
# mlrt_W2x(clip c, int "noise", int "scale", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", bool "preprocess")
###
## Parameters ##
#---------------
# c: Input clip.
# Must be in RGB 32-bit planar format or in Gray 32-bit format (model=0).
#---------------
# noise (default -1): Denoise level.
# Large value means strong denoise effect, -1 - no effect.
# Must be between -1..3.
#---------------
# scale (default 2): Upscale ratio.
# Must be either 1, 2, or 4.
#---------------
# tiles, tilesize (default not specified): Tiles size.
# If specified as single int, the values is used for both tile width and tile height.
# If specified as array, the first element is referred to tile width.
# If tilesize is specified, tiles doesn't have effect.
#---------------
# overlap_w, overlap_h (default overlap_w: 8 (model 0-4) / 4 (model 5-6)): Overlap width and overlap height of the tiles, respectively.
# overlap_h default: overlap_w.
#---------------
# model (default 6): What model to be used.
# Folder "models" must be in the same location as mlrt_xxx.dll.
/*
0: "anime_style_art"
1: "anime_style_art_rgb"
2: "photo"
3: "upconv_7_anime_style_art_rgb"
4: "upconv_7_photo"
5: "upresnet10"
6: "cunet" (not supported by ort(cuda))
7: "swin_unet_art" (not supported by ncnn)
8: "swin_unet_photo" (not supported by ncnn, ort(cuda/dml))
9: "swin_unet_photo_v2" (not supported by ncnn, ort(cuda/dml))
10: "swin_unet_art_scan" (not supported by ncnn, ort(cuda/dml))
*/
#---------------
# backend (default "ncnn"): What backend to be used (ncnn, ov, or ort) and its tuning parameters.
# It can be specified as single string.
# If specified as array of strings, the first element must be the backend type and the rest one are the tuning parameters.
# For example, backend=["ncnn",  "fp16=true"] - backend type is "ncnn" and fp16 mode is True.
#---------------
# preprocess (default True): Whether to upscale 2x before feeding the image to the models.


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


Function mlrt_W2x(clip c, int "noise", int "scale", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", bool "preprocess")
{
    noise = Default(noise, -1)
    scale = Default(scale, 2)
    model = Default(model, 6)
    Assert(model >= 0 && model <= 10, "mlrt_W2x: model must be 0..10")

    overlap_w = Default(overlap_w, Select(model, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4))
    overlap_h = Default(overlap_h, overlap_w)
    preprocess = Default(preprocess, True)
    backend = backend_define(backend, "mlrt_W2x")

    Assert(ComponentSize(c) == 4, "mlrt_W2x: clip must be in 32-bit planar format.")
    Assert(noise >= -1 && noise <= 3, "mlrt_W2x: noise must be -1, 0, 1, 2, or 3.")
    Assert(scale == 1 || scale == 2 || scale == 4, "mlrt_W2x: scale must be 1, 2, or 4.")
    if (backend[0] == "ncnn" && model > 6)
    {
        Assert(false, "mlrt_W2x: not supported model by ncnn.")
    }

    w2x_models = Select(model,
        \ "anime_style_art",
        \ "anime_style_art_rgb",
        \ "photo",
        \ "upconv_7_anime_style_art_rgb",
        \ "upconv_7_photo",
        \ "upresnet10",
        \ "cunet",
        \ "swin_unet_art",
        \ "swin_unet_photo",
        \ "swin_unet_photo_v2",
        \ "swin_unet_art_scan")

    Assert(!(model == 0 && noise == 0), "mlrt_W2x: anime_style_art model does not support noise reduction level 0.")
    if (model < 7 && scale == 4)
    {
        Assert(false, "mlrt_W2x: scale must be 1 or 2.")
    }

    if (model == 0)
    {
        Assert(NumComponents(c) == 1, "mlrt_W2x: clip must be of GRAY color family.")
    }
    else
    {
        Assert(IsPlanarRGB(c), "mlrt_W2x: clip must be of planar RGB color family.")
    }

    multiple = (model == 6) ? 4 : 1
    width_ = Width(c)

    if (preprocess && (model == 0 || model == 1 || model == 2))
    {
        # emulating cv2.resize(interpolation=cv2.INTER_CUBIC)
        c = z_ConvertFormat(c, BitLShiftL(width_, 1), BitLShiftL(height_, 1), filter_param_a=0, filter_param_b=0.75)
    }

    l = 0
    r = 0
    t = 0
    b = 0

    if (model == 6 && scale == 2)
    {
        bh = 4 - ((Width(c) - 1) % 4 + 1)
        bv = 4 - ((Height(c) - 1) % 4 + 1)
        l = BitRShiftL(bh, 1) + 4
        r = bh - l + 4
        t = BitRShiftL(bv, 1) + 4
        b = bv - t + 4
        AddBorders(c, l, t, r, b)
        c = FillBorders(left=l, top=t, right=r, bottom=b, mode=2)
    }

    tile_overlap = calc_tilesize(Width(c), Height(c), multiple, overlap_w, overlap_h, tiles, tilesize)

    Assert(!(tile_overlap[0] % multiple != 0 || tile_overlap[1] % multiple != 0), "mlrt_W2x: tile size must be divisible by " + String(multiple) + "(" + String(tile_overlap[0]) + ", " + String(tile_overlap[1]) + ").")

    folder_path = "waifu2x/" + w2x_models + "/"

    if (model == 0 || model == 1 || model == 2)
    {
        model_name = (noise == -1) ? "scale2.0x_model.onnx" : ("noise" + String(noise) + "_model.onnx")
    }
    else if (model == 3 || model == 4 || model == 5)
    {
        model_name = (noise == -1) ? "scale2.0x_model.onnx" : ("noise" + String(noise) + "_scale2.0x_model.onnx")
    }
    else if (model == 6)
    {
        scale_name = (scale == 1) ? "" : "scale2.0x_"

        model_name = (noise == -1) ? "scale2.0x_model.onnx" : ("noise" + String(noise) + "_" + scale_name + "model.onnx")
    }
    else if (model == 7)
    {
        Assert(noise != -1 && scale != 1, "mlrt_W2x: swin_unet model for noise=-1 and scale=1 does not exist.")

        scale_name = (scale > 1) ? (scale > 2) ? "scale4x" : "scale2x" : ""

        model_name = (noise > -1) ? (scale == 1) ? ("noise" + String(noise) + ".onnx") : ("noise" + String(noise) + "_" + scale_name + ".onnx") : (scale_name + ".onnx")
    }
    else
    {
        model_name = (noise == -1) ? "scale4x.onnx" : ("noise" + String(noise) + "_scale4x.onnx")
    }

    backend_defaults = backend_defaults(backend)

    c = Expr(c, "x 0 1 clip")
    c = Eval("mlrt_" + backend[0] + "(c, network_path=folder_path+model_name, overlap_w=tile_overlap[2], overlap_h=tile_overlap[3], tilesize_w=tile_overlap[0], tilesize_h=tile_overlap[1], builtin=true" + backend_defaults + ")")

    if (model < 8 && scale == 1 && (Width(c) / width_ == 2))
    {
        # emulating cv2.resize(interpolation=cv2.INTER_CUBIC)
        # cr: @AkarinVS
        c = fmtc_resample(c, scale=0.5, kernel="impulse", impulse=[-0.1875, 1.375, -0.1875], kovrspl=2)
    }
    else if (model > 7 && scale < 4)
    {
        c = z_ConvertFormat(c, width=BitRShiftL(Width(c) * scale, 2), height=BitRShiftL(Height(c) * scale, 2), filter_param_a=0, filter_param_b=0.5)
    }

    if (model == 6 && scale == 2)
    {
        c = Crop(c, BitLShiftL(l, 1), BitLShiftL(t, 1), -BitLShiftL(r, 1), -BitLShiftL(b, 1))
    }

    return c
}

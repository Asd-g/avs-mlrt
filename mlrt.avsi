### A wrapper of avs-mlrt for all bundled models.


### Requirements
# avs_mlrt
# avsresize - for mlrt_W2x(model=0/1/2, preprocess=true), mlrt_W2x(model=8, scale < 4)
# fmtconv - for mlrt_W2x(model=6, scale=1) and mlrt_RealESRGAN(scale=...)
# FillBorders - only for mlrt_W2x(model=6, scale=2)


### Usage ###
###
# mlrt_W2x(clip c, int "noise", int "scale", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", bool "preprocess")
#
# mlrt_DPIR(clip c, val "strength", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend")
#
# mlrt_RealESRGAN/mlrt_RealESRGANv2(clip c, val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", float "scale")
#
# mlrt_CUGAN(clip c, int "noise", int "scale", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", val "backend", int "version", bool "conformance")
#
# mlrt_SAFA(clip c, val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", int "adaptive")
###
## Parameters ##
#---------------
# c: Input clip.
# Must be in RGB 32-bit planar format or in Gray 32-bit format (mlrt_W2x(model=0), mlrt_DPIR(model=0/2)).
#---------------
# tiles, tilesize (default not specified): Tiles size.
# If specified as single int, the values is used for both tile width and tile height.
# If specified as array, the first element is referred to tile width.
# If tilesize is specified, tiles doesn't have effect.
#---------------
# overlap_w, overlap_h: Overlap width and overlap height of the tiles, respectively.
# mlrt_W2x overlap_w default: 8 (model 0-4), 4 (model 5-6).
# mlrt_DPIR overlap_w default: 0.
# mlrt_RealESRGAN overlap_w default: 8.
# mlrt_CUGAN overlap_w default: 4.
# overlap_h default: overlap_w.
#---------------
# backend (default "ncnn"): What backend to be used (ncnn, ov, or ort) and its tuning parameters.
# It can be specified as single string.
# If specified as array of strings, the first element must be the backend type and the rest one are the tuning parameters.
# For example, backend=["ncnn",  "fp16=true"] - backend type is "ncnn" and fp16 mode is True.
#---------------
# noise (mlrt_W2x, mlrt_CUGAN) (default -1): Denoise level.
# Large value means strong denoise effect, -1 - no effect.
# Must be between -1..3.
#---------------
# scale (mlrt_W2x, mlrt_RealESRGAN, mlrt_CUGAN): Upscale ratio.
# mlrt_W2x: default 2; must be either 1, 2, or 4.
# mlrt_RealESRGAN default: not specified.
# mlrt_CUGAN: default 2; must be 2, 3, or 4.
#---------------
# model (mlrt_W2x, mlrt_DPIR, mlrt_RealESRGAN, mlrt_SAFA): What model to be used.
# Folder "models" must be in the same location as mlrt_xxx.dll.
/*
mlrt_W2x:
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
Default 6.
*/
/*
mlrt_DPIR:
0: "drunet_gray"
1: "drunet_color"
2: "drunet_deblocking_grayscale"
3: "drunet_deblocking_color"
Default 0.
*/
/*
mlrt_RealESRGAN:
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
Default 0.
*/
/*
mlrt_SAFA:
0: v0.1 (not supported by ncnn)
1: v0.2 (not supported by ncnn)
2: v0.3 (not supported by ncnn)
3: v0.4 (not supported by ncnn)
*/
#---------------
# preprocess (mlrt_W2x) (default True): Whether to upscale 2x before feeding the image to the models.
#---------------
# strength (mlrt_DPIR) (default 5.0): Strength of denoising.
# It may be specified as clip (histogram mask).
#---------------
# version (mlrt_CUGAN) (default 1).
# 1: legacy
# 2: pro (only models for "noise" in [-1, 0, 3] and "scale" in [2, 3] are published currently)
#---------------
# conformance (mlrt_CUGAN) (default True): Currently specifies dynamic range compression for cugan-pro.
#---------------
# adaptive (mlrt_SAFA) (default 0).
# 0: non_adaptive
# 1: adaptive1x - use adaptive path only at 1x scale
# 2: adaptive - use adaptive path at 1x, 1/2x and 1/4x scales, proposed algorithm


### Version: 1.1.4


### Changelog ###
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
    Assert(backend[0] == "ov", "mlrt_SAFA: backend must be ov.")
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

Function calc_tilesize(int width, int height, int multiple, int overlap_w, int overlap_h, val "tiles", val "tilesize")
{
    if (!Defined(tilesize))
    {
        if (!Defined(tiles))
        {
            overlap_w = 0
            overlap_h = 0
            tile_w = width
            tile_h = height
        }
        else
        {
            tile_w = calc_size(width, IsArray(tiles) ? tiles[0] : tiles, overlap_w, multiple)
            tile_h = calc_size(height, IsArray(tiles) ? tiles[1] : tiles, overlap_h, multiple)
        }
    }
    else
    {
        tile_w = IsArray(tilesize) ? tilesize[0] : tilesize
        tile_h = IsArray(tilesize) ? tilesize[1] : tilesize
    }

    return [tile_w, tile_h, overlap_w, overlap_h]
}

Function calc_size(int width, int tiles, int overlap, int multiple)
{
    return Ceil((width + 2 * overlap * (tiles - 1)) / (tiles * multiple)) * multiple
}

Function backend_define(val "backend", string "function_name")
{
    backend_defined = Defined(backend) ? IsArray(backend) ? backend : [backend] : ["ncnn"]
    if (backend_defined[0] != "ncnn" && backend_defined[0] != "ov" && backend_defined[0] != "ort")
    {
        return Assert(false, function_name + ": not supported backend.")
    }
    else
    {
        return backend_defined
    }
}

Function backend_defaults(string_array backend)
{
    backend_string = (ArraySize(backend) == 1) ? "" : (", " + backend[1])

    for (i = 2, ArraySize(backend) - 1)
    {
        backend_string = backend_string + ", " + backend[i]
    }

    if (backend[0] == "ncnn")
    {
        /*
        basic performance tuning:
        set fp16 = True (on modern GPUs)
        increase num_streams
        */

        if (FindStr(backend_string, "fp16") == 0)
        {
            backend_string = backend_string + ", fp16=true"
        }
        if (FindStr(backend_string, "device_id") == 0)
        {
            backend_string = backend_string + ", device_id=0"
        }
        if (FindStr(backend_string, "num_streams") == 0)
        {
            backend_string = backend_string + ", num_streams=1"
        }
    }
    else if (backend[0] == "ov")
    {
        if (FindStr(backend_string, "device") == 0 || FindStr(backend_string, """device="CPU"""") > 0)
        {
            /*
            CPU basic performance tuning:
            ENFORCE_BF16=YES (on Zen4)
            increase num_streams
            */

            if (FindStr(backend_string, "config") == 0)
            {
                backend_string = backend_string + """, config="CPU_THROUGHPUT_STREAMS=1 CPU_BIND_THREAD=YES CPU_THREADS_NUM=0""""
            }
            else
            {
                if (FindStr(backend_string, "CPU_THROUGHPUT_STREAMS") == 0)
                {
                    beginning = MidStr(backend_string, FindStr(backend_string, "config"))

                    i = 9
                    while(MidStr(beginning, i, 1) != """"""")
                    {
                        i = i + 1
                    }

                    result = MidStr(beginning, 1, i-1) + " CPU_THROUGHPUT_STREAMS=1" + """""""
                    backend_string = ReplaceStr(backend_string, MidStr(beginning, 1, i), result)
                }
                if (FindStr(backend_string, "CPU_BIND_THREAD") == 0)
                {
                    beginning = MidStr(backend_string, FindStr(backend_string, "config"))

                    i = 9
                    while(MidStr(beginning, i, 1) != """"""")
                    {
                        i = i + 1
                    }

                    result = MidStr(beginning, 1, i-1) + " CPU_BIND_THREAD=YES" + """""""
                    backend_string = ReplaceStr(backend_string, MidStr(beginning, 1, i), result)
                }
                if (FindStr(backend_string, "CPU_THREADS_NUM") == 0)
                {
                    beginning = MidStr(backend_string, FindStr(backend_string, "config"))

                    i = 9
                    while(MidStr(beginning, i, 1) != """"""")
                    {
                        i = i + 1
                    }

                    result = MidStr(beginning, 1, i-1) + " CPU_THREADS_NUM=0" + """""""
                    backend_string = ReplaceStr(backend_string, MidStr(beginning, 1, i), result)
                }
            }
            if (FindStr(backend_string, "fp16") == 0)
            {
                backend_string = backend_string + ", fp16=false"
            }
            if (FindStr(backend_string, "fp16_blacklist_ops") == 0)
            {
                backend_string = backend_string + """, fp16_blacklist_ops="""""
            }
        }
        else
        {
            /*
            GPU basic performance tuning:
            set fp16 = True
            increase num_streams
            */

            if (FindStr(backend_string, "config") == 0)
            {
                backend_string = backend_string + """, config="GPU_THROUGHPUT_STREAMS=1""""
            }
            else
            {
                if (FindStr(backend_string, "GPU_THROUGHPUT_STREAMS") == 0)
                {
                    beginning = MidStr(backend_string, FindStr(backend_string, "config"))

                    i = 9
                    while(MidStr(beginning, i, 1) != """"""")
                    {
                        i = i + 1
                    }

                    result = MidStr(beginning, 1, i-1) + " GPU_THROUGHPUT_STREAMS=1" + """""""
                    backend_string = ReplaceStr(backend_string, MidStr(beginning, 1, i), result)
                }
            }
            if (FindStr(backend_string, "fp16") == 0)
            {
                backend_string = backend_string + ", fp16=false"
            }
            if (FindStr(backend_string, "fp16_blacklist_ops") == 0)
            {
                backend_string = backend_string + """, fp16_blacklist_ops="""""
            }
        }
    }

    return backend_string
}

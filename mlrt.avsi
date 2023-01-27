### A wrapper of avs_mlrt for all bundled models.


### Requirements - avs_mlrt, avsresize, fmtconv.


### Usage ###
###
# mlrt_W2x(clip c, int "noise", int "scale", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", bool "preprocess", string "backend_args")
#
# mlrt_DPIR(clip c, val "strength", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", string "backend_args")
#
# mlrt_RealESRGAN/mlrt_RealESRGANv2(clip c, val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", float "scale", string "backend_args")
#
# mlrt_CUGAN(clip c, int "noise", int "scale", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", val "backend", int "version", bool "conformance", string "backend_args")
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
# backend (default "ncnn"): What backend to be used (currently only ncnn) and its tuning parameters.
# It can be specified as single string.
# If specified as array of strings, the first element must be the backend type and the rest one are the tuning parameters.
# For example, backend=["ncnn",  "fp16=true"] - backend type is "ncnn" and fp16 mode is True.
#---------------
# backend_args (default not specified): Additional backend arguments.
#---------------
# noise (mlrt_W2x, mlrt_CUGAN) (default -1): Denoise level.
# Large value means strong denoise effect, -1 - no effect.
# Must be between -1..3.
#---------------
# scale (mlrt_W2x, mlrt_RealESRGAN, mlrt_CUGAN): Upscale ratio.
# mlrt_W2x: default 2; must be either 1 or 2.
# mlrt_RealESRGAN default: not specified.
# mlrt_CUGAN: default 2; must be 2, 3, or 4.
#---------------
# model (mlrt_W2x, mlrt_DPIR, mlrt_RealESRGAN): What model to be used.
/*
mlrt_W2x:
0: "anime_style_art"
1: "anime_style_art_rgb"
2: "photo"
3: "upconv_7_anime_style_art_rgb"
4: "upconv_7_photo"
5: "upresnet10"
6: "cunet"
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
Default 0.
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


### Version: 1.0.0


### Changelog ###
#---------------
# Initial version.


Function mlrt_W2x(clip c, int "noise", int "scale", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", bool "preprocess", string "backend_args")
{
    noise = Default(noise, -1)
    scale = Default(scale, 2)
    model = Default(model, 6)
    overlap_w = Default(overlap_w, Select(model, 8, 8, 8, 8, 8, 4, 4))
    overlap_h = Default(overlap_h, overlap_w)
    preprocess = Default(preprocess, True)
    backend_args = Defined(backend_args) ? (", " + backend_args) : ""
    backend = backend_def(backend)

    Assert(ComponentSize(c) == 4, "mlrt_W2x: clip must be in 32-bit planar format.")
    Assert(noise >= -1 && noise <= 3, "mlrt_W2x: noise must be -1, 0, 1, 2, or 3.")
    Assert(scale == 1 || scale == 2, "mlrt_W2x: scale must be 1 or 2.")
    Assert(model >= 0 && model <= 6, "mlrt_W2x: model must be 0, 1, 2, 3, 4, 5, or 6.")
    Assert(backend[0] == "ncnn", "mlrt_W2x: backend must be ncnn.")

    w2x_models = Select(model, "anime_style_art", "anime_style_art_rgb", "photo", "upconv_7_anime_style_art_rgb", "upconv_7_photo", "upresnet10", "cunet")

    Assert(!(model == 0 && noise == 0), "mlrt_W2x: anime_style_art model does not support noise reduction level 0.")

    if (model == 0)
    {
        Assert(NumComponents(c) == 1, "mlrt_W2x: clip must be of GRAY color family.")
    }
    else
    {
        Assert(IsPlanarRGB(c), "mlrt_W2x: clip must be of planar RGB color family.")
    }

    multiple = (model == 6) ? 4 : 1
    width = Width(c)
    height = Height(c)

    if (preprocess && (model == 0 || model == 1 || model == 2))
    {
        # emulating cv2.resize(interpolation=cv2.INTER_CUBIC)
        c = z_ConvertFormat(c, width * 2, height * 2, filter_param_a=0, filter_param_b=0.75)
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
    else
    {
        scale_name = (scale == 1) ? NOP() : "scale2.0x_"

        model_name = (noise == -1) ? "scale2.0x_model.onnx" : ("noise" + String(noise) + "_" + scale_name + "model.onnx")
    }

    backend_defaults = backend_defaults(backend)

    c = Eval("mlrt_" + backend[0] + "(c, network_path=folder_path+model_name, overlap_w=tile_overlap[2], overlap_h=tile_overlap[3], tilesize_w=tile_overlap[0], tilesize_h=tile_overlap[1], builtin=true, " + backend_defaults + backend_args + ")")

    if (scale == 1 && (Width(c) / width == 2))
    {
        # emulating cv2.resize(interpolation=cv2.INTER_CUBIC)
        # cr: @AkarinVS
        c = fmtc_resample(c, scale=0.5, kernel="impulse", impulse=[-0.1875, 1.375, -0.1875], kovrspl=2)
    }

    return c
}

Function mlrt_DPIR(clip c, val "strength", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", string "backend_args")
{
    model = Default(model, 0)
    overlap_w = Default(overlap_w, 0)
    overlap_h = Default(overlap_h, overlap_w)
    backend_args = Defined(backend_args) ? (", " + backend_args) : ""
    backend = backend_def(backend)

    Assert(ComponentSize(c) == 4, "mlrt_DPIR: clip must be in 32-bit planar format.")
    Assert(model >= 0 && model <= 3, "mlrt_DPIR: model must be 0, 1, 2, or 3.")
    Assert(backend[0] == "ncnn", "mlrt_DPIR: backend must be ncnn.")

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

    return Eval("mlrt_" + backend[0] + "([c, strength], network_path=network_path, overlap_w=tile_overlap[2], overlap_h=tile_overlap[3], tilesize_w=tile_overlap[0], tilesize_h=tile_overlap[1], builtin=true, " + backend_defaults + backend_args + ")")
}

Function mlrt_RealESRGAN(clip c, val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", float "scale", string "backend_args")
{
    model = Default(model, 0)
    overlap_w = Default(overlap_w, 8)
    overlap_h = Default(overlap_h, overlap_w)
    backend_args = Defined(backend_args) ? (", " + backend_args) : ""
    backend = backend_def(backend)

    Assert(ComponentSize(c) == 4, "mlrt_RealESRGAN: clip must be in 32-bit planar format.")
    Assert(IsPlanarRGB(c), "mlrt_RealESRGAN: clip must be of planar RGB color family.")
    Assert(model >= 0 && model <= 2, "mlrt_RealESRGAN: model must be 0, 1, or 2.")
    Assert(backend[0] == "ncnn", "mlrt_RealESRGAN: backend must be ncnn.")

    # v2, v2, v3 4x
    realesrgan_models = Select(model, "animevideo-xsx2", "animevideo-xsx4", "animevideov3")

    multiple = 1

    tile_overlap = calc_tilesize(Width(c), Height(c), multiple, overlap_w, overlap_h, tiles, tilesize)

    network_path = (model == 0 || model == 1) ? ("RealESRGANv2/" + "RealESRGANv2-" + realesrgan_models + ".onnx") : \
    ("RealESRGANv2/" + "realesr-" + realesrgan_models + ".onnx")

    backend_defaults = backend_defaults(backend)

    clip_org = c
    c = Eval("mlrt_" + backend[0] + "(c, network_path=network_path, overlap_w=tile_overlap[2], overlap_h=tile_overlap[3], tilesize_w=tile_overlap[0], tilesize_h=tile_overlap[1], builtin=true, " + backend_defaults + backend_args + ")")

    if (Defined(scale))
    {
        scale_h = Width(c) / Width(clip_org)
        scale_v = Height(c) / Height(clip_org)

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

Function mlrt_RealESRGANv2(clip c, val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", val "backend", float "scale", string "backend_args")
{
    return mlrt_RealESRGAN(c, tiles, tilesize, overlap_w, overlap_h, model, backend, scale, backend_args)
}

Function mlrt_CUGAN(clip c, int "noise", int "scale", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", val "backend", int "version", bool "conformance", string "backend_args")
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
    backend_args = Defined(backend_args) ? (", " + backend_args) : ""
    backend = backend_def(backend)

    Assert(ComponentSize(c) == 4, "mlrt_CUGAN: clip must be in 32-bit planar format.")
    Assert(IsPlanarRGB(c), "mlrt_CUGAN: clip must be of planar RGB color family.")
    Assert(noise >= -1 && noise <= 3, "mlrt_CUGAN: noise must be -1, 0, 1, 2, or 3.")
    Assert(scale == 2 || scale == 3 || scale == 4, "mlrt_CUGAN: scale must be 2, 3, or 4.")
    Assert(backend[0] == "ncnn", "mlrt_CUGAN: backend must be ncnn.")
    Assert(version == 1 || version == 2, "mlrt_CUGAN: version must be 1 (legacy) or 2 (pro)")
    Assert(!(scale != 2 && (noise == 1 || noise == 2)), "mlrt_CUGAN: scale=" + String(scale) + " model.\nmlrt_CUGAN does not support noise reduction level " + String(noise) + ".")

    multiple = 2

    width = Width(c)
    height = Height(c)

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

    c = Eval("mlrt_" + backend[0] + "(c, network_path=folder_path+model_name, overlap_w=tile_overlap[2], overlap_h=tile_overlap[3], tilesize_w=tile_overlap[0], tilesize_h=tile_overlap[1], builtin=true, " + backend_defaults + backend_args + ")")

    # https://github.com/bilibili/ailab/blob/e102bef22384c629f82552dbec3d6b5bab125639/Real-CUGAN/upcunet_v3.py#L269
    if (conformance && version == 2)
    {
        c = Expr(c, "x 0.15 - 0.7 /")
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

Function ncnn_defaults(bool "fp16", int "device_id", int "num_streams")
{
    /*
    basic performance tuning:
    set fp16 = True (on modern GPUs)
    increase num_streams
    */

    fp16 = Default(fp16, True)
    device_id = Default(device_id, 0)
    num_streams = Default(num_streams, 1)

    return "fp16=" + String(fp16) + ", " + "device_id=" + String(device_id) + ", " + "num_streams=" + String(num_streams)
}

Function backend_def(val "backend")
{
    return Defined(backend) ? IsArray(backend) ? backend : \
    [backend] : ["ncnn"]
}

Function backend_defaults(string_array backend)
{
    backend_string = backend_format(backend)

    if (backend[0] == "ncnn")
    {
        return eval("ncnn_defaults(" + backend_string + ")")
    }
}

Function backend_format(string_array backend)
{
    backend_string = ""

    for (i = 1, ArraySize(backend) - 1)
    {
        backend_string = (i == 1) ? backend[1] : (backend_string + ", " + backend[i])
    }

    return backend_string
}

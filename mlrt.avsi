Function mlrt_w2x(clip c, int "noise", int "scale", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", string "backend", bool "preprocess", bool "fp16", string "backend_args")
{
    noise = Default(noise, -1)
    scale = Default(scale, 2)
    model = Default(model, 6)
    backend = Default(backend, "mlrt_ncnn")

    Assert(backend == "mlrt_ncnn", "mlrt_w2x: backend must be mlrt_ncnn.")

    preprocess = Default(preprocess, true)

    Assert(ComponentSize(c) == 4, "mlrt_w2x: clip must be in 32-bit planar format.")
    Assert(noise >= -1 && noise <= 3, "mlrt_w2x: noise must be -1, 0, 1, 2, or 3.")
    Assert(scale == 1 || scale == 2, "mlrt_w2x: scale must be 1 or 2.")
    Assert(model >= 0 && model <= 6, "mlrt_w2x: model must be 0, 1, 2, 3, 4, 5, or 6.")

    w2x_models = Select(model, "anime_style_art", "anime_style_art_rgb", "photo", "upconv_7_anime_style_art_rgb", "upconv_7_photo", "upresnet10", "cunet")

    Assert(!(model == 0 && noise == 0), "mlrt_w2x: anime_style_art model does not support noise reduction level 0.")

    if (model == 0)
    {
        Assert(NumComponents(c) == 1, "mlrt_w2x: clip must be of GRAY color family.")
    }
    else
    {
        Assert(IsPlanarRGB(c), "mlrt_w2x: clip must be of planar RGB color family.")
    }

    overlap_w = Default(overlap_w, Select(model, 8, 8, 8, 8, 8, 4, 4))
    overlap_h = Default(overlap_h, overlap_w)

    multiple = (model == 6) ? 4 : 1
    width = Width(c)
    height = Height(c)

    if (preprocess && (model == 0 || model == 1 || model == 2))
    {
        # emulating cv2.resize(interpolation=cv2.INTER_CUBIC)
        c = z_ConvertFormat(c, width * 2, height * 2, filter_param_a=0, filter_param_b=0.75)
    }

    tile_overlap = calc_tilesize(Width(c), Height(c), multiple, overlap_w, overlap_h, tiles, tilesize)

    Assert(!(tile_overlap[0] % multiple != 0 || tile_overlap[1] % multiple != 0), "mlrt_w2x: tile size must be divisible by " + String(multiple) + "(" + String(tile_overlap[0]) + ", " + String(tile_overlap[1]) + ").")

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

    fp16 = Default(fp16, true)
    backend_args = Defined(backend_args) ? (", " + backend_args) : ""

    c = Eval(backend + "(c, network_path=folder_path+model_name, overlap_w=tile_overlap[2], overlap_h=tile_overlap[3], tilesize_w=tile_overlap[0], tilesize_h=tile_overlap[1], builtin=true, fp16=fp16" + backend_args + ")")

    if (scale == 1 && (Width(c) / width == 2))
    {
        c = fmtc_resample(c, scale=0.5, kernel="impulse", impulse=[-0.1875, 1.375, -0.1875], kovrspl=2)
    }

    return c
}

Function mlrt_dpir(clip c, val "strength", val "tiles", val "tilesize", int "overlap_w", int "overlap_h", int "model", string "backend", bool "fp16", string "backend_args")
{
    Assert(ComponentSize(c) == 4, "mlrt_dpir: clip must be in 32-bit planar format.")

    model = Default(model, 0)
    Assert(model >= 0 && model <= 3, "mlrt_dpir: model must be 0, 1, 2, or 3.")

    dpir_models = Select(model, "drunet_gray", "drunet_color", "drunet_deblocking_grayscale", "drunet_deblocking_color")

    Assert(!((model == 0 || model == 2) && NumComponents(c) > 1), "mlrt_dpir: clip must be of GRAY color family.")
    Assert(!((model == 1 || model == 3) && !IsPlanarRGB(c)), "mlrt_dpir: clip must be of RGB color family.")

    if (!Defined(strength))
    {
        strength = 5.0
    }
    if (IsClip(strength))
    {
        Assert(NumComponents(strength) == 1, "mlrt_dpir: strength must be of GRAY color family.")
        Assert(!(Width(strength) != Width(c) || Height(strength) != Height(c)), "mlrt_dpir: strength must be of the same size as clip.")
        Assert(FrameCount(strength) == FrameCount(c), "mlrt_dpir: strength must be of the same length as clip.")

        strength = Expr(strength, "x 255 /")
    }
    else
    {
        strength = strength
        Assert(IsFloat(strength) || IsInt(strength), "mlrt_dpir: strength must be a float or a clip.")

        strength = strength / 255.0
        strength = KillAudio(BlankClip(c, pixel_type="y32", colors=[strength, strength, strength]))
        strength = Loop(Trim(strength, 0, -1), FrameCount(c))
    }

    overlap_w = Default(overlap_w, 0)
    overlap_h = Default(overlap_h, overlap_w)

    multiple = 8

    tile_overlap = calc_tilesize(Width(c), Height(c), multiple, overlap_w, overlap_h, tiles, tilesize)

    Assert(!(tile_overlap[0] % multiple != 0 || tile_overlap[1] % multiple != 0), "mlrt_dpir: tile size must be divisible by " + String(multiple) + "(" + String(tile_overlap[0]) + ", " + String(tile_overlap[1]) + ").")

    backend = Default(backend, "mlrt_ncnn")

    Assert(backend == "mlrt_ncnn", "mlrt_dpir: backend must be mlrt_ncnn.")

    network_path = "dpir/" + dpir_models + ".onnx"

    fp16 = Default(fp16, true)
    backend_args = Defined(backend_args) ? (", " + backend_args) : ""

    return Eval(backend + "([c, strength], network_path=network_path, overlap_w=tile_overlap[2], overlap_h=tile_overlap[3], tilesize_w=tile_overlap[0], tilesize_h=tile_overlap[1], builtin=true, fp16=fp16" + backend_args + ")")
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

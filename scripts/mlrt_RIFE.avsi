### A avs-mlrt wrapper of RIFE.
### This is a port of the https://github.com/AmusementClub/vs-mlrt/blob/master/scripts/vsmlrt.py


### Requirements
# avs_mlrt
# mlrt_common.avsi
# avsresize
# FrameSel


### Usage ###
###
# Function mlrt_RIFE(clip c, int implementation, int "factor_num", int "factor_den", int "fps_num", int "fps_den", int "model", val "backend", bool "ensemble", bool "sc", float "sc_threshold")
###
## Parameters ##
#---------------
# c: Input clip.
# Must be in RGB 32-bit planar format.
#---------------
# implementation: The onnx implementation.
# Must be between 1 and 2.
# 1: The video width and height must be mod32.
# 2: It has internal padding.
#---------------
# factor_num, factor_den: Factor of target frame rate.
# Default: factor_num = 2, factor_den = 1.
# factor_num must be at least 2x factor_den.
#---------------
# fps_num, fps_den: Target frame rate.
# Default: not specified.
# If these are specified they supersedes factor_num and factor_den parameter.
#---------------
# model (default 3): What model to be used.
# For implementation=1: models must be placed in folder "models/rife_mlrt/rife" where the root folder "models" must be in the same location as mlrt_xxx.dll.
# For implementation=2: models must be placed in folder "models/rife_mlrt/rife_v2" where the root folder "models" must be in the same location as mlrt_xxx.dll.
/*
0: "v4.0"
1: "v4.2"
2: "v4.3"
3: "v4.4"
4: "v4.5"
5: "v4.6"
6: "v4.7"
7: "v4.8"
8: "v4.9"
9: "v4.10"
10: "v4.11"
11: "v4.12"
12: "v4.12_lite"
13: "v4.13"
14: "v4.13_lite"
15: "v4.14"
16: "v4.14_lite"
17: "v4.15"
18: "v4.15_lite"
19: "v4.16_lite"
20: "v4.17"
21: "v4.17_lite"
22: "v4.18"
23: "v4.19"
24: "v4.20"
25: "v4.21"
26: "v4.22"
27: "v4.22_lite"
*/
#---------------
# backend (default "ov"): What backend to be used (ov, or ort) and its tuning parameters.
# It can be specified as single string.
# If specified as array of strings, the first element must be the backend type and the rest one are the tuning parameters.
# ncnn is not supported.
# For example, backend=["ov",  "fp16=true"] - backend type is "ov" and fp16 mode is True.
#---------------
# ensemble (default False): Whether to use speed or quality oriented version of the model.
# False: speed oriented.
# True: quality oriented.
#---------------
# sc (default False): Avoid interpolating frames over scene changes.
# It doesn't have effect when factor_num % factor_den != 0.
#---------------
# sc_threshold (default 0.1): Threshold to determine whether the current frame and the next one are end/beginning of scene.
# Must be between 0.0..1.0.


### Version: 1.1.0


### Changelog ###
#---------------
# 1.1.0
# Added parameters fps_num and fps_den.
#---------------
# 1.0.2
# Added support for model v4.22_lite.
# Fixed model names.
#---------------
# 1.0.1
# Added support for models: v4_17, v4_17_lite, v4_18, v4_19, v4_20, v4_21, v4_22.
#---------------
# 1.0.0
# Initial version.


Function mlrt_RIFE(clip c, int implementation, int "factor_num", int "factor_den", int "fps_num", int "fps_den", int "model", val "backend", bool "ensemble", bool "sc", float "sc_threshold")
{
    factor_num = Default(factor_num, 2)
    factor_den = Default(factor_den, 1)
    model = Default(model, 3)
    backend = backend_define(Default(backend, "ov"), "mlrt_RIFE")
    ensemble = Default(ensemble, False)
    sc = Default(sc, False)
    sc_threshold = Default(sc_threshold, 0.1)

    Assert(ComponentSize(c) == 4, "mlrt_RIFE: clip must be in 32-bit planar format.")
    Assert(IsPlanarRGB(c), "mlrt_RIFE: clip must be of planar RGB color family.")
    Assert(implementation != 1 || implementation != 2, "mlrt_RIFE: implementation must be either 1 or 2.")
    Assert(model >= 0 && model <= 27, "mlrt_RIFE: model must be between 0..27.")
    Assert(backend[0] == "ov" || backend[0] == "ort", "mlrt_RIFE: backend must be ov or ort.")
    Assert(sc_threshold >= 0.0 && sc_threshold <= 1.0, "mlrt_RIFE: sc_threshold must be between 0.0..1.0.")

    rife_models = Select(model,
        \ "v4.0",
        \ "v4.2",
        \ "v4.3",
        \ "v4.4",
        \ "v4.5",
        \ "v4.6",
        \ "v4.7",
        \ "v4.8",
        \ "v4.9",
        \ "v4.10",
        \ "v4.11",
        \ "v4.12",
        \ "v4.12_lite",
        \ "v4.13",
        \ "v4.13_lite",
        \ "v4.14",
        \ "v4.14_lite",
        \ "v4.15",
        \ "v4.15_lite",
        \ "v4.16_lite",
        \ "v4.17",
        \ "v4.17_lite",
        \ "v4.18",
        \ "v4.19",
        \ "v4.20",
        \ "v4.21",
        \ "v4.22",
        \ "v4.22_lite")

    multi = (Defined(fps_num) && Defined(fps_den)) ?
        \ ((Float(fps_num) / fps_den) / FrameRate(c)) :
        \ (Float(factor_num) / factor_den)

    c = Expr(c, "x 0 1 clip")

    if (Frac(multi) == 0.0)
    {
        multi = factor_num / factor_den
        Assert(multi >= 2, "mlrt_RIFE: factor_num must be at least 2x factor_den.")

        c_array = [c]
        multi_minus_one = multi - 1
        for (i = 2, multi_minus_one)
        {
            c_array = ArrayAdd(c_array, c)
        }

        initial = Interleave(c_array)

        terminal = DuplicateFrame(c, FrameCount(c) - 1)
        terminal = Trim(terminal, 1, 0)

        terminal_array = [terminal]
        for (i = 2, multi_minus_one)
        {
            terminal_array = ArrayAdd(terminal_array, terminal)
        }

        terminal = Interleave(terminal_array)

        timepoint_array = []
        for (i = 1, multi_minus_one)
        {
            timepoint_array = ArrayAdd(timepoint_array, BlankClip(clip=c, colors=[Float(i) / multi], pixel_type="y32"))
        }

        timepoint = Interleave(timepoint_array)

        output0 = RIFEMerge(c, initial, terminal, timepoint, implementation, rife_models, ensemble, backend)

        if (sc)
        {
            y_clip = z_ConvertFormat(output0, pixel_type="y32", colorspace_op="rgb:709:709:f=>709:709:709:l")
            output0 = ScriptClip(output0, function [y_clip, initial, sc_threshold] ()
            {
                if (YDifferenceToNext(y_clip) > sc_threshold)
                {
                    return FrameSel(initial, current_frame)
                }
            })
        }

        if (multi == 2)
        {
            res = Interleave(c, output0)
        }
        else
        {
            res_array = [c]
            for (i = 0, multi_minus_one - 1)
            {
                res_array = ArrayAdd(res_array, SelectEvery(output0, multi - 1, i))
            }

            res = Interleave(res_array)
        }

        return (FrameRateNumerator(c) != 0 && FrameRateDenominator(c) != 0) ?
            \ AssumeFPS(res, FrameRateNumerator(c) * multi, FrameRateDenominator(c))
            \ : res
    }
    else
    {
        src_fps_num = FrameRateNumerator(c)
        src_fps_den = FrameRateDenominator(c)
        src_fps = (src_fps_num == 0 || src_fps_den == 0) ? 1.0 : FrameRate(c)
        dst_fps = src_fps * multi
        src_frames = FrameCount(c)
        dst_frames = Min(Int(src_frames * multi), Int(Pow(2, 31)) - 1)

        duration_rel = src_fps / dst_fps
        dst_duration = ContinuedNumerator(duration_rel, limit=50000)
        src_duration = ContinuedDenominator(duration_rel, limit=50000)

        temp = BlankClip(clip=c, length=dst_frames)

        left_clip = ScriptClip(temp, function [c, src_duration, dst_duration] ()
        {
            return FrameSel(c, dst_duration * current_frame / src_duration)
        })

        right_clip = ScriptClip(temp, function [c, src_duration, dst_duration] ()
        {
            return FrameSel(c, dst_duration * current_frame / src_duration + 1)
        })

        temp_gray = BlankClip(clip=temp, pixel_type="y32")
        tp_clip = ScriptClip(temp_gray, function [c, src_duration, dst_duration] ()
        {
            current_time = dst_duration * current_frame
            return BlankClip(clip=last, colors=Float(current_time - src_duration * (current_time / src_duration)) / src_duration)
        })

        output0 = RIFEMerge(c, left_clip, right_clip, tp_clip, implementation, rife_models, ensemble, backend)

        y_clip = z_ConvertFormat(output0, pixel_type="y32", colorspace_op="rgb:709:709:f=>709:709:709:l")
        res = ScriptClip(output0, function [left_clip, src_duration, dst_duration, src_frames, y_clip, sc_threshold] ()
        {
            current_time = dst_duration * current_frame
            if (current_time % src_duration == 0 || (current_time / src_duration) + 1 >= src_frames || YDifferenceToNext(y_clip) > sc_threshold)
            {
                return left_clip
            }
        })

        return (src_fps_num != 0 && src_fps_den != 0) ?
            \ AssumeFPS(res, ContinuedNumerator(dst_fps, limit=50000), ContinuedDenominator(dst_fps, limit=50000))
            \ : res
    }
}

Function RIFEMerge(clip c, clip clipa, clip clipb, clip clipc, int implementation, string rife_models, bool ensemble, string_array backend)
{
    if (implementation == 1)
    {
        Assert(!(Width(c) % 32 != 0 || Height(c) % 32 != 0), "mlrt_RIFE: width/height must be divisible by " + String(32) + "(" + String(Width(c)) + ", " + String(Height(c)) + ").")

        empty = BlankClip(clip=clipa, pixel_type="y32")
        horizontal = Expr(empty, "sx 2 * width 1 - / 1 -")
        vertical = Expr(empty, "sy 2 * height 1 - / 1 -")

        multiplier_h = BlankClip(clip=clipa, pixel_type="y32", colors=2.0/(Width(clipa) - 1))
        multiplier_v = BlankClip(clip=clipa, pixel_type="y32", colors=2.0/(Height(clipa) - 1))

        network_path = "rife_mlrt/rife/" + "rife_" + rife_models + ((ensemble) ? "_ensemble" : "") + ".onnx"

        backend_defaults = backend_defaults(backend)

        return Eval("mlrt_" + backend[0] + "([clipa, clipb, clipc, horizontal, vertical, multiplier_h, multiplier_v], network_path=network_path, builtin=true" + backend_defaults + ")")
    }
    else
    {
        network_path = "rife_mlrt/rife_v2/" + "rife_" + rife_models + ((ensemble) ? "_ensemble" : "") + ".onnx"

        backend_defaults = backend_defaults(backend)

        return Eval("mlrt_" + backend[0] + "([clipa, clipb, clipc], network_path=network_path, builtin=true" + backend_defaults + ")")
    }
}

### Helper functions for the mlrt_xxx.avsi scripts.


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

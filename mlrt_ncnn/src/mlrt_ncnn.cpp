#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#if !(__cpp_lib_atomic_wait)
#include <chrono>
#include <thread>
#endif

#include "avisynth_c.h"
#include "boost/dll/runtime_symbol_info.hpp"

// ncnn
#include <net.h>
#include <gpu.h>

#include "onnx/common/version.h"
#include "onnx2ncnn.h"


extern std::variant<std::string, ONNX_NAMESPACE::ModelProto> loadONNX(
    const std::string& path,
    const unsigned CP,
    int64_t tile_w,
    int64_t tile_h,
    bool path_is_serialization
) noexcept;


[[nodiscard]]
static std::optional<std::string> checkNodes(
    const std::vector<const AVS_VideoInfo*>& vis
) noexcept
{

    for (const auto& vi : vis)
    {
        if (avs_component_size(vi) < 4)
            return "expects clip with type fp32";
        if (vi->width != vis[0]->width || vi->height != vis[0]->height)
            return "dimensions of clips mismatch";
        if (vi->num_frames != vis[0]->num_frames)
            return "number of frames mismatch";
        if (!avs_is_rgb(vi) && !avs_is_y(vi))
            return "clip must not be sub-sampled";
    }

    return {};
}

struct TicketSemaphore
{
    std::atomic<intptr_t> ticket{};
    std::atomic<intptr_t> current{};

    void acquire() noexcept
    {
        intptr_t tk{ ticket.fetch_add(1, std::memory_order_acquire) };
        while (true)
        {
            intptr_t curr{ current.load(std::memory_order_acquire) };
            if (tk <= curr)
            {
                return;
            }
#if __cpp_lib_atomic_wait
            current.wait(curr, std::memory_order::relaxed);
#else // __cpp_lib_atomic_wait
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(10ms);
#endif // __cpp_lib_atomic_wait
        }
    }

    void release() noexcept
    {
        current.fetch_add(1, std::memory_order_release);
#if __cpp_lib_atomic_wait
        current.notify_all();
#endif // __cpp_lib_atomic_wait
    }
};

// per-stream context
struct Resource
{
    std::unique_ptr<ncnn::VkCompute> cmd;
    ncnn::VkAllocator* blob_vkallocator;
    ncnn::VkAllocator* staging_vkallocator;
    ncnn::Mat h_src_fp32;
    ncnn::Mat h_src;
    ncnn::VkMat d_src;
    ncnn::VkMat d_dst;
    ncnn::Mat h_dst;
    ncnn::Mat h_dst_fp32;
};

static std::atomic<int> num_plugin_instances{};

struct avsNcnnData
{
    std::vector<AVS_Clip*> nodes;

    int overlap_w, overlap_h;

    int in_tile_c, in_tile_w, in_tile_h;
    int out_tile_c, out_tile_w, out_tile_h;

    bool fp16;

    std::vector<Resource> resources;
    std::vector<int> tickets;
    std::mutex ticket_lock;
    TicketSemaphore semaphore;

    ncnn::VulkanDevice* device; // ncnn caches device allocations in a global variable
    ncnn::Net net;
    int input_index;
    int output_index;

    int acquire() noexcept
    {
        semaphore.acquire();
        {
            std::lock_guard<std::mutex> lock(ticket_lock);
            int ticket{ tickets.back() };
            tickets.pop_back();
            return ticket;
        }
    }

    void release(int ticket) noexcept
    {
        {
            std::lock_guard<std::mutex> lock(ticket_lock);
            tickets.push_back(ticket);
        }
        semaphore.release();
    }

    std::string err;
};


static AVS_VideoFrame* AVSC_CC get_frame_mlrt_ncnn(AVS_FilterInfo* fi, int n)
{

    avsNcnnData* d{ static_cast<avsNcnnData*>(fi->user_data) };

    std::vector<const AVS_VideoInfo*> in_vis(std::size(d->nodes) + 1);
    in_vis[0] = avs_get_video_info(fi->child);
    for (int i{ 1 }; const auto & node : d->nodes)
    {
        in_vis[i] = avs_get_video_info(node);
        ++i;
    }

    std::vector<AVS_VideoFrame*> src_frames(std::size(in_vis));
    src_frames[0] = avs_get_frame(fi->child, n);
    for (int i{ 1 }; const auto & node : d->nodes)
    {
        src_frames[i] = avs_get_frame(node, n);
        ++i;
    }

    for (int i{ 0 }; i < std::size(src_frames); ++i)
    {
        if (!src_frames[i])
            return nullptr;
    }

    auto src_stride{ avs_get_pitch(src_frames.front()) };
    auto src_width{ avs_get_row_size(src_frames.front()) / avs_component_size(in_vis.front()) };
    auto src_height{ avs_get_height(src_frames.front()) };

    AVS_VideoFrame* dst_frame{ avs_new_video_frame_p(fi->env, &fi->vi, src_frames.front()) };
    auto dst_stride{ avs_get_pitch(dst_frame) };

    auto ticket{ d->acquire() };
    Resource& resource = d->resources[ticket];

    std::array<int64_t, 4> src_tile_shape{ 1, d->in_tile_c, d->in_tile_h, d->in_tile_w };
    auto src_tile_h{ src_tile_shape[2] };
    auto src_tile_w{ src_tile_shape[3] };
    auto src_tile_w_bytes{ src_tile_w * avs_component_size(in_vis.front()) };

    const int planes_r[3] = { AVS_PLANAR_R, AVS_PLANAR_G, AVS_PLANAR_B };
    const int plane_y{ AVS_PLANAR_Y };
    const int* planes{ (avs_is_rgb(in_vis.front())) ? planes_r : &plane_y };

    std::vector<const uint8_t*> src_ptrs(src_tile_shape[1]);
    int num_planes_total{};
    for (unsigned i{ 0 }; i < std::size(src_frames); ++i)
    {
        for (int j{ 0 }; j < avs_num_components(in_vis[i]); ++j, ++num_planes_total)
            src_ptrs[num_planes_total] = avs_get_read_ptr_p(src_frames[i], planes[j]);
    }

    auto step_w{ src_tile_w - 2 * d->overlap_w };
    auto step_h{ src_tile_h - 2 * d->overlap_h };

    std::array<int64_t, 4> dst_tile_shape{ 1, d->out_tile_c, d->out_tile_h, d->out_tile_w };
    auto dst_tile_h{ dst_tile_shape[2] };
    auto dst_tile_w{ dst_tile_shape[3] };
    auto dst_tile_w_bytes{ dst_tile_w * avs_component_size(&fi->vi) };
    auto dst_planes{ dst_tile_shape[1] };
    uint8_t* dst_ptrs[3]{};
    for (int i{ 0 }; i < dst_planes; ++i)
        dst_ptrs[i] = avs_get_write_ptr_p(dst_frame, planes[i]);

    auto h_scale{ dst_tile_h / src_tile_h };
    auto w_scale{ dst_tile_w / src_tile_w };

    const auto set_error{ [&](const std::string& error_message)
    {
        using namespace std::string_literals;

        d->release(ticket);

        avs_release_video_frame(dst_frame);

        for (const auto& frame : src_frames)
        {
            avs_release_video_frame(frame);
        }

        d->err = "mlrt_ncnn: "s + error_message;
        fi->error = d->err.c_str();

        return nullptr;
    } };

    ncnn::Option opt{ d->net.opt };
    opt.blob_vkallocator = resource.blob_vkallocator;
    opt.workspace_vkallocator = resource.blob_vkallocator;
    opt.staging_vkallocator = resource.staging_vkallocator;

    int y{ 0 };
    while (true)
    {
        int y_crop_start{ (y == 0) ? 0 : d->overlap_h };
        int y_crop_end{ (y == src_height - src_tile_h) ? 0 : d->overlap_h };

        int x{ 0 };
        while (true)
        {
            int x_crop_start{ (x == 0) ? 0 : d->overlap_w };
            int x_crop_end{ (x == src_width - src_tile_w) ? 0 : d->overlap_w };

            {
                auto input_buffer{ reinterpret_cast<uint8_t*>(d->fp16 ? resource.h_src_fp32.data : resource.h_src.data) };

                // assumes the pitches of ncnn::Mat to be
                // (cstep * elemsize, w * h * elemsize, h * elemsize)
                for (const auto& _src_ptr : src_ptrs)
                {
                    const uint8_t* src_ptr{ _src_ptr +
                        y * src_stride + x * avs_component_size(in_vis.front())
                    };

                    {
                        avs_bit_blt(fi->env,
                            input_buffer, src_tile_w_bytes,
                            src_ptr, src_stride,
                            src_tile_w_bytes, src_tile_h
                        );
                        input_buffer += resource.h_src.cstep * sizeof(float);
                    }
                }
            }

            if (d->fp16)
                ncnn::cast_float32_to_float16(resource.h_src_fp32, resource.h_src);

            resource.cmd->record_clone(resource.h_src, resource.d_src, opt);

            {
                auto extractor{ d->net.create_extractor() };
                extractor.set_blob_vkallocator(resource.blob_vkallocator);
                extractor.set_workspace_vkallocator(resource.blob_vkallocator);
                extractor.set_staging_vkallocator(resource.staging_vkallocator);
                extractor.input(d->input_index, resource.d_src);
                extractor.extract(d->output_index, resource.d_dst, *resource.cmd);
            }

            resource.cmd->record_clone(resource.d_dst, resource.h_dst, opt);
            if (resource.cmd->submit_and_wait() != 0)
            {
                resource.cmd->reset();
                return set_error("inference failed");
            }
            if (resource.cmd->reset() != 0)
                return set_error("cmd reset failed");

            if (d->fp16)
                ncnn::cast_float16_to_float32(resource.h_dst, resource.h_dst_fp32);

            {
                auto output_buffer{ reinterpret_cast<uint8_t*>(d->fp16 ? resource.h_dst_fp32.data : resource.h_dst.data) };

                for (int plane{ 0 }; plane < dst_planes; ++plane)
                {
                    auto dst_ptr{ (dst_ptrs[plane] +
                        h_scale * y * dst_stride + w_scale * x * avs_component_size(&fi->vi)
                        ) };

                    {
                        avs_bit_blt(fi->env,
                            dst_ptr + (y_crop_start * dst_stride + x_crop_start * avs_component_size(&fi->vi)),
                            dst_stride,
                            output_buffer + (y_crop_start * dst_tile_w_bytes + x_crop_start * avs_component_size(&fi->vi)),
                            dst_tile_w_bytes,
                            dst_tile_w_bytes - (x_crop_start + x_crop_end) * avs_component_size(&fi->vi),
                            dst_tile_h - (y_crop_start + y_crop_end)
                        );

                        output_buffer += resource.h_dst.cstep * sizeof(float);
                    }
                }
            }

            if (x + src_tile_w == src_width)
                break;

            x = (std::min)(x + step_w, src_width - src_tile_w);
        }

        if (y + src_tile_h == src_height)
            break;

        y = (std::min)(y + step_h, src_height - src_tile_h);
    }

    d->release(ticket);

    for (const auto& frame : src_frames)
        avs_release_video_frame(frame);

    return dst_frame;
}


static void AVSC_CC free_mlrt_ncnn(AVS_FilterInfo* fi)
{
    avsNcnnData* d{ static_cast<avsNcnnData*>(fi->user_data) };

    for (const auto& node : d->nodes)
        avs_release_clip(node);

    for (const auto& resource : d->resources)
    {
        d->device->reclaim_blob_allocator(resource.blob_vkallocator);
        d->device->reclaim_staging_allocator(resource.staging_vkallocator);
    }

    delete d;

    if (--num_plugin_instances == 0)
        ncnn::destroy_gpu_instance();
}


static int AVSC_CC set_cache_hints_mlrt_ncnn(AVS_FilterInfo* fi, int cachehints, int frame_range)
{
    return cachehints == AVS_CACHE_GET_MTMODE ? 2 : 0;
}


static AVS_Value AVSC_CC Create_mlrt_ncnn(AVS_ScriptEnvironment* env, AVS_Value args, void* param)
{
    enum { Clips, Network_path, Overlap_w, Overlap_h, Tilesize_w, Tilesize_h, Device_id, Num_streams, Builtin, Builtindir, Fp16, Path_is_serialization, List_gpu };

    avsNcnnData* d{ new avsNcnnData() };
    ++num_plugin_instances;

    AVS_FilterInfo* fi;
    AVS_Clip* clip{ avs_new_c_filter(env, &fi, *avs_as_array(avs_array_elt(args, Clips)), 1) };

    if (avs_defined(avs_array_elt(args, List_gpu)) ? avs_as_bool(avs_array_elt(args, List_gpu)) : 0)
    {
        for (auto i{ 0 }; i < ncnn::get_gpu_count(); ++i)
            d->err += std::to_string(i) + ": " + ncnn::get_gpu_info(i).device_name() + "\n";

        AVS_Value cl{ avs_new_value_clip(clip) };
        AVS_Value args_[2]{ cl, avs_new_value_string(d->err.c_str()) };
        AVS_Value v{ avs_invoke(fi->env, "Text", avs_new_value_array(args_, 2), 0) };

        avs_release_value(cl);
        avs_release_clip(clip);

        if (--num_plugin_instances == 0)
            ncnn::destroy_gpu_instance();

        return v;
    }

    const int num_nodes{ avs_array_size(avs_array_elt(args, Clips)) };
    d->nodes.reserve(num_nodes - 1);
    for (int i{ 0 }; i < num_nodes - 1; ++i)
        d->nodes.emplace_back(avs_take_clip(*(avs_as_array(avs_array_elt(args, Clips)) + (i + 1)), env));

    auto set_error{ [&](const std::string& error_message)
    {
        using namespace std::string_literals;

        avs_release_clip(clip);

        for (const auto& node : d->nodes)
            avs_release_clip(node);

        if (--num_plugin_instances == 0)
            ncnn::destroy_gpu_instance();

        d->err = "mlrt_ncnn: "s + error_message;

        return avs_new_value_error(d->err.c_str());
    } };

    if (avs_check_version(env, 10))
        return set_error("AviSynth+ version must be r3928 or later.");

    std::vector<const AVS_VideoInfo*> in_vis(num_nodes);
    in_vis[0] = &fi->vi;
    for (int i{ 1 }; const auto & node : d->nodes)
    {
        in_vis[i] = avs_get_video_info(node);
        ++i;
    }

    if (auto err{ checkNodes(in_vis) }; err.has_value())
        return set_error(err.value());

    const int device_id{ avs_defined(avs_array_elt(args, Device_id)) ? (avs_as_int(avs_array_elt(args, Device_id))) : 0 };

    d->overlap_w = avs_defined(avs_array_elt(args, Overlap_w)) ? (avs_as_int(avs_array_elt(args, Overlap_w))) : -4525;
    d->overlap_h = avs_defined(avs_array_elt(args, Overlap_h)) ? (avs_as_int(avs_array_elt(args, Overlap_h))) : -4525;
    if (d->overlap_w != -4525)
    {
        if (d->overlap_h == -4525)
            d->overlap_h = d->overlap_w;

        if (d->overlap_w < 0)
            return set_error("overlap_w must be non-negative");
        if (d->overlap_h < 0)
            return set_error("overlap_h must be non-negative");
    }
    else
    {
        d->overlap_w = 0;

        if (d->overlap_h == -4525)
            d->overlap_h = 0;

        if (d->overlap_h < 0)
            return set_error("overlap_h must be non-negative");
    }

    int tile_w{ avs_defined(avs_array_elt(args, Tilesize_w)) ? (avs_as_int(avs_array_elt(args, Tilesize_w))) : -4525 };
    int tile_h{ avs_defined(avs_array_elt(args, Tilesize_h)) ? (avs_as_int(avs_array_elt(args, Tilesize_h))) : -4525 };
    if (tile_w != -4525)
    { // manual specification triggered
        if (tile_h == -4525)
            tile_h = tile_w;
    }
    else
    {
        if (tile_h == -4525)
        {
            if (d->overlap_w != 0)
                return set_error("tilesize_w must be specified");

            if (d->overlap_h != 0)
                return set_error("tilesize_h must be specified");
        }

        // set tile size to video dimensions
        tile_w = fi->vi.width;
        tile_h = fi->vi.height;
    }
    if (tile_w - 2 * d->overlap_w <= 0)
        return set_error("overlap_w too large");
    if (tile_h - 2 * d->overlap_h <= 0)
        return set_error("overlap_h too large");

    const int num_streams{ avs_defined(avs_array_elt(args, Num_streams)) ? (avs_as_int(avs_array_elt(args, Num_streams))) : 1 };
    if (num_streams <= 0)
        return set_error("num_streams must be positive");

    d->semaphore.current.store(num_streams - 1, std::memory_order_relaxed);
    d->tickets.reserve(num_streams);
    for (int i{ 0 }; i < num_streams; ++i)
        d->tickets.push_back(i);

    d->fp16 = !!avs_defined(avs_array_elt(args, Fp16)) ? (avs_as_bool(avs_array_elt(args, Fp16))) : false;

    const bool path_is_serialization{ static_cast<bool>(!!avs_defined(avs_array_elt(args, Path_is_serialization)) ? (avs_as_bool(avs_array_elt(args, Path_is_serialization))) : false) };
    std::string network_path{ avs_as_string(avs_array_elt(args, Network_path)) };
    if (!network_path.size())
        return set_error("network_path must be specified");

    const bool builtin{ static_cast<bool>(!!avs_defined(avs_array_elt(args, Builtin)) ? (avs_as_bool(avs_array_elt(args, Builtin))) : true) };
    if (builtin)
    {
        std::string modeldir{ avs_defined(avs_array_elt(args, Builtindir)) ? (avs_as_string(avs_array_elt(args, Builtindir))) : "models" };
        network_path = boost::dll::this_line_location().parent_path().generic_string() + "/" + modeldir + "/" + network_path;
    }

    auto result{ loadONNX(network_path, 0, tile_w, tile_h, path_is_serialization) };
    if (std::holds_alternative<std::string>(result))
    {
        result = loadONNX(network_path, 65001, tile_w, tile_h, path_is_serialization);
        if (std::holds_alternative<std::string>(result))
            return set_error(std::get<std::string>(result));
    }

    auto onnx_model{ std::move(std::get<ONNX_NAMESPACE::ModelProto>(result)) };
    {
        auto int64ToIntS{ [&](int64_t i)
        {
            if (i > INT_MAX)
                return INT_MAX;
            else if (i < INT_MIN)
                return INT_MIN;
            else return (int)i;
        } };

        const auto& input_shape = onnx_model.graph().input(0).type().tensor_type().shape();
        d->in_tile_c = int64ToIntS(input_shape.dim(1).dim_value());
        d->in_tile_h = int64ToIntS(input_shape.dim(2).dim_value());
        d->in_tile_w = int64ToIntS(input_shape.dim(3).dim_value());

        const auto& output_shape = onnx_model.graph().output(0).type().tensor_type().shape();
        d->out_tile_c = int64ToIntS(output_shape.dim(1).dim_value());
        d->out_tile_h = int64ToIntS(output_shape.dim(2).dim_value());
        d->out_tile_w = int64ToIntS(output_shape.dim(3).dim_value());
    }

    fi->vi.width *= d->out_tile_w / d->in_tile_w;
    fi->vi.height *= d->out_tile_h / d->in_tile_h;
    if (d->out_tile_c == 1)
        fi->vi.pixel_type = AVS_CS_Y32;
    if (d->out_tile_c == 3)
        fi->vi.pixel_type = AVS_CS_RGBPS;

    auto ncnn_result{ onnx2ncnn(onnx_model) };
    if (!ncnn_result.has_value())
        return set_error("onnx2ncnn failed");

    const auto& [ncnn_param, ncnn_model_bin] = ncnn_result.value();

    auto aligned_free{ [&](void* ptr)
    {
#ifdef _MSC_VER 
        _aligned_free(ptr);
#else 
        free(ptr);
#endif
    }
    };

    // ncnn related code
    if (auto device{ ncnn::get_gpu_device(device_id) }; device != nullptr)
        d->device = device;
    else
    {
        aligned_free(ncnn_param);
        aligned_free(ncnn_model_bin);
        return set_error("get_gpu_device failed");
    }

    d->net.opt.num_threads = 1;
    d->net.opt.use_vulkan_compute = true;
    d->net.opt.use_fp16_packed = d->fp16;
    d->net.opt.use_fp16_storage = d->fp16;
    d->net.opt.use_int8_storage = false;
    d->net.set_vulkan_device(d->device);
    if (d->net.load_param_mem(ncnn_param) != 0)
    {
        aligned_free(ncnn_param);
        aligned_free(ncnn_model_bin);
        return set_error("load param failed");
    }
    aligned_free(ncnn_param);
    // TODO: here returns the number of bytes read successfully
    d->net.load_model(ncnn_model_bin);
    aligned_free(ncnn_model_bin);

    d->input_index = d->net.input_indexes().front();
    d->output_index = d->net.output_indexes().front();

    size_t bps{ 4 };
    if (d->fp16)
        bps = 2;

    d->resources.resize(num_streams);
    for (auto& resource : d->resources)
    {
        resource.cmd = std::make_unique<ncnn::VkCompute>(d->device);
        resource.blob_vkallocator = d->device->acquire_blob_allocator();
        resource.staging_vkallocator = d->device->acquire_staging_allocator();
        resource.h_src.create(d->in_tile_w, d->in_tile_h, d->in_tile_c, bps);
        resource.d_src.create(d->in_tile_w, d->in_tile_h, d->in_tile_c, bps, resource.blob_vkallocator);
        resource.d_dst.create(d->out_tile_w, d->out_tile_h, d->out_tile_c, bps, resource.blob_vkallocator);
        resource.h_dst.create(d->out_tile_w, d->out_tile_h, d->out_tile_c, bps);
        if (d->fp16)
        {
            resource.h_src_fp32.create(d->in_tile_w, d->in_tile_h, d->in_tile_c, sizeof(float));
            resource.h_dst_fp32.create(d->out_tile_w, d->out_tile_h, d->out_tile_c, sizeof(float));
        }
    }

    AVS_Value v{ avs_new_value_clip(clip) };

    fi->user_data = reinterpret_cast<void*>(d);
    fi->get_frame = get_frame_mlrt_ncnn;
    fi->set_cache_hints = set_cache_hints_mlrt_ncnn;
    fi->free_filter = free_mlrt_ncnn;

    avs_release_clip(clip);

    return v;
}


const char* AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment* env)
{
    avs_add_function(env, "mlrt_ncnn", "c+[network_path]s[overlap_w]i[overlap_h]i[tilesize_w]i[tilesize_h]i[device_id]i[num_streams]i[builtin]b[builtindir]s[fp16]b[path_is_serialization]b[list_gpu]b", Create_mlrt_ncnn, 0);
    return "mlrt_ncnn";
}

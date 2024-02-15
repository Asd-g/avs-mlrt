#include <array>
#include <atomic>
#include <cstdint>
#include <ios>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <variant>
#include <vector>

#if not __cpp_lib_atomic_wait
#include <chrono>
#include <thread>
using namespace std::chrono_literals;
#endif

#include "onnx/common/version.h"
#include "onnx/onnx_pb.h"

#define NOMINMAX

#include "onnxruntime_c_api.h"

#include <cuda_runtime.h>

#include "dml_provider_factory.h"

#include "avisynth_c.h"
#include "boost/dll/runtime_symbol_info.hpp"

#ifdef _WIN32
#include <Windows.h>
#endif // _WIN32



extern std::variant<std::string, ONNX_NAMESPACE::ModelProto> loadONNX(
    const std::string& path,
    const unsigned CP,
    int64_t tile_w,
    int64_t tile_h,
    bool path_is_serialization
) noexcept;

extern void convert_float_to_float16(
    ONNX_NAMESPACE::ModelProto& model,
    bool force_fp16_initializers,
    const std::unordered_set<std::string>& op_block_list
) noexcept;

#define checkError(expr) do {                                                  \
    OrtStatusPtr __err{ expr };                                                \
    if (__err) {                                                               \
        const std::string message{ ortapi->GetErrorMessage(__err) };           \
        ortapi->ReleaseStatus(__err);                                          \
        return set_error("'"s + # expr + "' failed: " + message);              \
    }                                                                          \
} while(0)

#define checkCUDAError(expr) do {                                              \
    if (cudaError_t result = expr; result != cudaSuccess) {                    \
        const char * error_str{ cudaGetErrorString(result)};                   \
        return set_error("'"s + # expr + "' failed: " + error_str);            \
    }                                                                          \
} while(0)

using namespace std::string_literals;

static const OrtApi* ortapi{ nullptr };
static std::atomic<int64_t> logger_id{ 0 };
static std::mutex capture_lock;

// rename GridSample to com.microsoft::GridSample
// onnxruntime has support for CUDA-accelerated GridSample only in its own opset domain
static void rename(ONNX_NAMESPACE::ModelProto& model)
{
    constexpr auto ms_domain{ "com.microsoft" };

    bool has_ms_opset{ false };
    for (const auto& opset : model.opset_import())
    {
        if (opset.has_domain() && opset.domain() == ms_domain)
        {
            has_ms_opset = true;
            break;
        }
    }

    if (!has_ms_opset)
    {
        ONNX_NAMESPACE::OperatorSetIdProto opset_id;
        *opset_id.mutable_domain() = ms_domain;
        opset_id.set_version(1);
        *model.add_opset_import() = std::move(opset_id);
    }

    for (auto& node : *model.mutable_graph()->mutable_node())
    {
        if (node.has_op_type() && node.op_type() == "GridSample")
            *node.mutable_domain() = ms_domain;
    }
}

[[nodiscard]]
static std::optional<std::string> ortInit() noexcept
{
    static std::once_flag ort_init_flag;

    std::call_once(ort_init_flag, []()
    {
        auto p{ OrtGetApiBase() };
        if (p)
            ortapi = p->GetApi(ORT_API_VERSION);
    });

    if (ortapi)
        return {};
    else
        return "ONNX Runtime initialization failed";
}

[[nodiscard]]
static std::variant<std::string, std::array<int64_t, 4>> getShape(const OrtTensorTypeAndShapeInfo* tensor_info) noexcept
{
    const auto set_error{ [](const std::string& error_message)
    {
        return error_message;
    } };

    std::array<int64_t, 4> shape;
    checkError(ortapi->GetDimensions(tensor_info, std::data(shape), std::size(shape)));

    return shape;
}

[[nodiscard]]
static std::variant<std::string, std::array<int64_t, 4>> getShape(const OrtSession* session, bool input) noexcept
{
    const auto set_error{ [](const std::string& error_message)
    {
        return error_message;
    } };

    OrtTypeInfo* typeinfo;
    if (input)
        checkError(ortapi->SessionGetInputTypeInfo(session, 0, &typeinfo));
    else
        checkError(ortapi->SessionGetOutputTypeInfo(session, 0, &typeinfo));

    const OrtTensorTypeAndShapeInfo* tensor_info;
    checkError(ortapi->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

    auto maybe_shape{ getShape(tensor_info) };
    ortapi->ReleaseTypeInfo(typeinfo);

    if (std::holds_alternative<std::string>(maybe_shape))
        return set_error(std::get<std::string>(maybe_shape));

    return std::get<std::array<int64_t, 4>>(maybe_shape);
}

[[nodiscard]]
static std::optional<std::string> checkIOInfo(const OrtTypeInfo* info, bool is_output) noexcept
{
    const auto set_error{ [](const std::string& error_message)
    {
        return error_message;
    } };

    const OrtTensorTypeAndShapeInfo* tensor_info;
    checkError(ortapi->CastTypeInfoToTensorInfo(info, &tensor_info));

    ONNXTensorElementDataType element_type;
    checkError(ortapi->GetTensorElementType(tensor_info, &element_type));

    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        return set_error("expects network IO with type fp32");

    size_t num_dims;
    checkError(ortapi->GetDimensionsCount(tensor_info, &num_dims));
    if (num_dims != 4)
        return set_error("expects network with 4-D IO");

    auto maybe_shape{ getShape(tensor_info) };
    if (std::holds_alternative<std::string>(maybe_shape))
        return set_error(std::get<std::string>(maybe_shape));

    auto shape{ std::get<std::array<int64_t, 4>>(maybe_shape) };
    if (shape[0] != 1)
        return set_error("batch size of network must be 1");

    if (is_output)
    {
        int64_t out_channels{ shape[1] };
        if (out_channels != 1 && out_channels != 3)
            return "output dimensions must be 1 or 3";
    }

    return {};
}

[[nodiscard]]
static std::optional<std::string> checkSession(const OrtSession* session) noexcept
{
    const auto set_error{ [](const std::string& error_message)
    {
        return error_message;
    } };

    size_t num_inputs;
    checkError(ortapi->SessionGetInputCount(session, &num_inputs));

    if (num_inputs != 1)
        return set_error("network input count must be 1, got " + std::to_string(num_inputs));

    OrtTypeInfo* input_type_info;
    checkError(ortapi->SessionGetInputTypeInfo(session, 0, &input_type_info));

    if (auto err{ checkIOInfo(input_type_info, false) }; err.has_value())
        return set_error(err.value());

    ortapi->ReleaseTypeInfo(input_type_info);

    size_t num_outputs;
    checkError(ortapi->SessionGetOutputCount(session, &num_outputs));

    if (num_outputs != 1)
        return "network output count must be 1, got " + std::to_string(num_outputs);

    OrtTypeInfo* output_type_info;
    checkError(ortapi->SessionGetOutputTypeInfo(session, 0, &output_type_info));

    if (auto err{ checkIOInfo(output_type_info, true) }; err.has_value())
        return set_error(err.value());

    ortapi->ReleaseTypeInfo(output_type_info);

    return {};
}

[[nodiscard]]
static std::optional<std::string> checkNodesAndNetwork(const OrtSession* session,
    const std::unique_ptr<const AVS_VideoInfo* []>& vis, const int num_nodes) noexcept
{
    const auto set_error{ [](const std::string& error_message)
    {
        return error_message;
    } };

    OrtTypeInfo* input_type_info;
    checkError(ortapi->SessionGetInputTypeInfo(session, 0, &input_type_info));

    const OrtTensorTypeAndShapeInfo* input_tensor_info;
    checkError(ortapi->CastTypeInfoToTensorInfo(input_type_info, &input_tensor_info));

    auto network_in_dims{ std::get<std::array<int64_t, 4>>(getShape(input_tensor_info)) };

    int network_in_channels{ static_cast<int>(network_in_dims[1]) };
    int num_planes{ 0 };
    for (int i{ 0 }; i < num_nodes; ++i)
        num_planes += avs_num_components(vis[i]);

    if (network_in_channels != num_planes)
        return set_error("expects " + std::to_string(network_in_channels) + " input planes");

    auto clip_in_height{ vis[0]->height };
    auto clip_in_width{ vis[0]->width };
    if (network_in_dims[2] > clip_in_height || network_in_dims[3] > clip_in_width)
        return set_error("tile size larger than clip dimension");

    ortapi->ReleaseTypeInfo(input_type_info);

    return {};
}

static void setDimensions(AVS_VideoInfo* vi, const std::array<int64_t, 4>& input_shape,
    const std::array<int64_t, 4>& output_shape) noexcept
{
    vi->height *= output_shape[2] / input_shape[2];
    vi->width *= output_shape[3] / input_shape[3];

    if (output_shape[1] == 1)
        vi->pixel_type = AVS_CS_Y32;
    else if (output_shape[1] == 3)
        vi->pixel_type = AVS_CS_RGBPS;
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
                return;
#if __cpp_lib_atomic_wait
            current.wait(curr, std::memory_order::relaxed);
#else // __cpp_lib_atomic_wait
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

enum class Backend
{
    UNKNOWN = -1,
    CPU = 0,
    CUDA = 1,
    DML = 2
};

struct CUDA_Resource_t
{
    uint8_t* h_data;
    uint8_t* d_data;
    size_t size;
};

// per-stream context
struct Resource
{
    OrtSession* session;
    OrtValue* input_tensor;
    OrtValue* output_tensor;
    OrtIoBinding* binding;
    char* input_name;
    char* output_name;

    cudaStream_t stream;
    CUDA_Resource_t input;
    CUDA_Resource_t output;
    bool require_replay;
};

struct ORTData
{
    std::unique_ptr<AVS_Clip* []> nodes;

    int overlap_w;
    int overlap_h;

    OrtEnv* environment;

    std::unique_ptr<Resource[]> resources;
    std::vector<int> tickets;
    std::mutex ticket_lock;
    TicketSemaphore semaphore;

    int acquire() noexcept
    {
        semaphore.acquire();
        {
            std::lock_guard<std::mutex> lock(ticket_lock);
            int ticket = tickets.back();
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
    int num_nodes;
    int num_streams;
    int device_id;

#ifdef _WIN32
    HMODULE dml_dll;
#endif // _WIN32

};

template <Backend backend>
static AVS_VideoFrame* AVSC_CC get_frame_mlrt_ort(AVS_FilterInfo* fi, int n)
{
    ORTData* d{ static_cast<ORTData*>(fi->user_data) };
    const int num_nodes{ d->num_nodes };

    std::unique_ptr<const AVS_VideoInfo* []> in_vis{ std::make_unique_for_overwrite<const AVS_VideoInfo * []>(num_nodes) };
    in_vis[0] = &fi->vi;

    std::unique_ptr<AVS_VideoFrame* []> src_frames{ std::make_unique_for_overwrite< AVS_VideoFrame * [] >(num_nodes) };
    src_frames[0] = avs_get_frame(fi->child, n);
    if (!src_frames[0])
        return nullptr;

    for (int i{ 1 }; i < num_nodes; ++i)
    {
        in_vis[i] = avs_get_video_info(d->nodes[i - 1]);
        src_frames[i] = avs_get_frame(d->nodes[i - 1], n);
        if (!src_frames[i])
        {
            while (--i >= 0)
                avs_release_frame(src_frames[i]);

            return nullptr;
        }
    }

    auto src_stride{ avs_get_pitch(src_frames[0]) };
    auto src_width{ avs_get_row_size(src_frames[0]) / avs_component_size(in_vis[0]) };
    auto src_height{ avs_get_height(src_frames[0]) };
    auto src_bytes{ avs_component_size(in_vis[0]) };

    auto ticket{ d->acquire() };
    Resource& resource{ d->resources[ticket] };

    auto src_tile_shape{ std::get<std::array<int64_t, 4>>(getShape(resource.session, true)) };
    auto src_tile_h{ src_tile_shape[2] };
    auto src_tile_w{ src_tile_shape[3] };
    auto src_tile_w_bytes{ src_tile_w * src_bytes };
    auto src_tile_bytes{ src_tile_h * src_tile_w_bytes };

    constexpr int planes_r[3] = { AVS_PLANAR_R, AVS_PLANAR_G, AVS_PLANAR_B };
    constexpr int plane_y{ AVS_PLANAR_Y };
    const int* planes{ (avs_is_rgb(in_vis[0])) ? planes_r : &plane_y };

    std::unique_ptr<const uint8_t* []> src_ptrs{ std::make_unique_for_overwrite<const uint8_t * []>(src_tile_shape[1]) };
    int num_planes_total{};
    for (int i{ 0 }; i < num_nodes; ++i)
    {
        for (int j{ 0 }; j < avs_num_components(in_vis[i]); ++j, ++num_planes_total)
            src_ptrs[num_planes_total] = avs_get_read_ptr_p(src_frames[i], planes[j]);
    }

    auto step_w{ src_tile_w - 2 * d->overlap_w };
    auto step_h{ src_tile_h - 2 * d->overlap_h };

    AVS_VideoFrame* dst_frame{ avs_new_video_frame_p(fi->env, &fi->vi, src_frames[0]) };

    auto dst_stride{ avs_get_pitch(dst_frame) };
    auto dst_bytes{ avs_component_size(&fi->vi) };
    auto dst_tile_shape{ std::get<std::array<int64_t, 4>>(getShape(resource.session, false)) };
    auto dst_tile_h{ dst_tile_shape[2] };
    auto dst_tile_w{ dst_tile_shape[3] };
    auto dst_tile_w_bytes{ dst_tile_w * dst_bytes };
    auto dst_tile_bytes{ dst_tile_h * dst_tile_w_bytes };
    auto dst_planes{ dst_tile_shape[1] };
    std::array<uint8_t*, 3> dst_ptrs{};
    for (int i{ 0 }; i < dst_planes; ++i)
        dst_ptrs[i] = avs_get_write_ptr_p(dst_frame, planes[i]);

    auto h_scale{ dst_tile_h / src_tile_h };
    auto w_scale{ dst_tile_w / src_tile_w };

    const auto set_error{ [&](const std::string& error_message)
    {
        d->release(ticket);

        avs_release_video_frame(dst_frame);

        for (int i{ 0 }; i < num_nodes; ++i)
            avs_release_video_frame(src_frames[i]);

        d->err = "mlrt_ort: " + error_message;
        fi->error = d->err.c_str();

        return nullptr;
    } };

    if constexpr (backend == Backend::CUDA)
        checkCUDAError(cudaSetDevice(d->device_id));

    int y{ 0 };
    while (true)
    {
        const int y_crop_start{ (y == 0) ? 0 : d->overlap_h };
        const int y_crop_end{ (y == src_height - src_tile_h) ? 0 : d->overlap_h };

        int x{ 0 };
        while (true)
        {
            const int x_crop_start{ (x == 0) ? 0 : d->overlap_w };
            const int x_crop_end{ (x == src_width - src_tile_w) ? 0 : d->overlap_w };

            {
                uint8_t* input_buffer;
                uint8_t* h_input_buffer{ resource.input.h_data };
                checkError(ortapi->GetTensorMutableData(
                    resource.input_tensor,
                    reinterpret_cast<void**>(&input_buffer)
                ));

                for (int i{ 0 }; i < src_tile_shape[1]; ++i)
                {
                    const uint8_t* src_ptr{ src_ptrs[i] + y * src_stride + x * src_bytes };

                    if constexpr (backend == Backend::CUDA)
                    {
                        avs_bit_blt(fi->env, h_input_buffer, src_tile_w_bytes, src_ptr, src_stride,
                            src_tile_w_bytes, src_tile_h);

                        h_input_buffer += src_tile_bytes;
                    }
                    else
                    {
                        avs_bit_blt(fi->env, input_buffer, src_tile_w_bytes, src_ptr, src_stride,
                            src_tile_w_bytes, src_tile_h);

                        input_buffer += src_tile_bytes;
                    }
                }
            }

            if constexpr (backend == Backend::CUDA)
            {
                checkCUDAError(cudaMemcpyAsync(
                    resource.input.d_data,
                    resource.input.h_data,
                    resource.input.size,
                    cudaMemcpyHostToDevice,
                    resource.stream
                ));

                // OrtCUDAProviderOptionsV2 disallows using custom user stream
                // and the inference is executed on a private non-blocking stream
                checkCUDAError(cudaStreamSynchronize(resource.stream));
            }

            if (resource.require_replay) [[unlikely]]
            {
                resource.require_replay = false;

                // runs it under a global lock
                // onnxruntime uses global-mode stream capture on a private stream
                // this lock prevents concurrent capture sequences in other threads
                //
                // note that this applies only to stream capture from the ort library
                // this fails when another plugin also uses global-mode stream capture
                std::lock_guard _{ capture_lock };
                checkError(ortapi->RunWithBinding(resource.session, nullptr, resource.binding));

                // onnxruntime replays the graph itself in CUDAExecutionProvider::OnRunEnd
            }
            else if constexpr (backend == Backend::CPU || backend == Backend::CUDA)
                checkError(ortapi->RunWithBinding(resource.session, nullptr, resource.binding));
            else
            {
                checkError(ortapi->Run(
                    resource.session,
                    nullptr,
                    &resource.input_name,
                    &resource.input_tensor,
                    1,
                    &resource.output_name,
                    1,
                    &resource.output_tensor
                ));
            }

            if constexpr (backend == Backend::CUDA)
            {
                checkCUDAError(cudaMemcpyAsync(
                    resource.output.h_data,
                    resource.output.d_data,
                    resource.output.size,
                    cudaMemcpyDeviceToHost,
                    resource.stream
                ));
                checkCUDAError(cudaStreamSynchronize(resource.stream));
            }

            {
                uint8_t* output_buffer;
                uint8_t* h_output_buffer = resource.output.h_data;
                checkError(ortapi->GetTensorMutableData(
                    resource.output_tensor,
                    reinterpret_cast<void**>(&output_buffer)
                ));

                for (int plane{ 0 }; plane < dst_planes; ++plane)
                {
                    auto dst_ptr{ (dst_ptrs[plane] + h_scale * y * dst_stride + w_scale * x * dst_bytes) };

                    if constexpr (backend == Backend::CUDA)
                    {
                        avs_bit_blt(fi->env, dst_ptr + (y_crop_start * dst_stride + x_crop_start * dst_bytes),
                            dst_stride,
                            h_output_buffer + (y_crop_start * dst_tile_w_bytes + x_crop_start * dst_bytes),
                            dst_tile_w_bytes,
                            dst_tile_w_bytes - (x_crop_start + x_crop_end) * dst_bytes,
                            dst_tile_h - (y_crop_start + y_crop_end));

                        h_output_buffer += dst_tile_bytes;
                    }
                    else
                    {
                        avs_bit_blt(fi->env, dst_ptr + (y_crop_start * dst_stride + x_crop_start * dst_bytes),
                            dst_stride,
                            output_buffer + (y_crop_start * dst_tile_w_bytes + x_crop_start * dst_bytes),
                            dst_tile_w_bytes,
                            dst_tile_w_bytes - (x_crop_start + x_crop_end) * dst_bytes,
                            dst_tile_h - (y_crop_start + y_crop_end));

                        output_buffer += dst_tile_bytes;
                    }
                }
            }

            if (x + src_tile_w == src_width)
                break;

            x = std::min(x + step_w, src_width - src_tile_w);
        }

        if (y + src_tile_h == src_height)
            break;

        y = std::min(y + step_h, src_height - src_tile_h);
    }

    d->release(ticket);

    for (int i{ 0 }; i < num_nodes; ++i)
        avs_release_video_frame(src_frames[i]);

    return dst_frame;
}

template <Backend backend>
static void AVSC_CC free_mlrt_ort(AVS_FilterInfo* fi)
{
    ORTData* d{ static_cast<ORTData*>(fi->user_data) };

    for (int i{ 0 }; i < d->num_nodes - 1; ++i)
        avs_release_clip(d->nodes[i]);

    for (int i{ 0 }; i < d->num_streams; ++i)
    {
        ortapi->ReleaseIoBinding(d->resources[i].binding);
        ortapi->ReleaseValue(d->resources[i].output_tensor);
        ortapi->ReleaseValue(d->resources[i].input_tensor);
        ortapi->ReleaseSession(d->resources[i].session);

        if constexpr (backend == Backend::CUDA)
        {
            cudaStreamDestroy(d->resources[i].stream);
            cudaFreeHost(d->resources[i].input.h_data);
            cudaFree(d->resources[i].input.d_data);
            cudaFreeHost(d->resources[i].output.h_data);
            cudaFree(d->resources[i].output.d_data);
        }
    }

    ortapi->ReleaseEnv(d->environment);

#ifdef _WIN32
    if constexpr (backend == Backend::DML)
        FreeLibrary(d->dml_dll);
#endif // _WIN32

    delete d;
}


static int AVSC_CC set_cache_hints_mlrt_ort(AVS_FilterInfo* fi, int cachehints, int frame_range)
{
    return cachehints == AVS_CACHE_GET_MTMODE ? 2 : 0;
}


static AVS_Value AVSC_CC Create_mlrt_ort(AVS_ScriptEnvironment* env, AVS_Value args, void* param)
{
    enum
    {
        Clips, Network_path, Overlap_w, Overlap_h, Tilesize_w, Tilesize_h, Provider, Device, Num_streams, Verbosity,
        Cudnn_benchmark, Builtin, Builtindir, Fp16, Path_is_serialization, Use_cuda_graph, Fp16_blacklist_ops
    };

    ORTData* d{ new ORTData() };

    AVS_FilterInfo* fi;
    AVS_Clip* clip{ avs_new_c_filter(env, &fi, *avs_as_array(avs_array_elt(args, Clips)), 1) };

    const int num_nodes{ avs_array_size(avs_array_elt(args, Clips)) };
    d->num_nodes = num_nodes;
    if (num_nodes > 1)
    {
        d->nodes = std::make_unique_for_overwrite<AVS_Clip * []>(num_nodes - 1);
        for (int i{ 1 }; i < num_nodes; ++i)
            d->nodes[i - 1] = avs_take_clip(*(avs_as_array(avs_array_elt(args, Clips)) + i), env);
    }

    auto set_error{ [&](const std::string& error_message)
    {
        avs_release_clip(clip);

        for (int i{ 1 }; i < num_nodes; ++i)
            avs_release_clip(d->nodes[i - 1]);

        d->err = "mlrt_ort: " + error_message;

        return avs_new_value_error(d->err.c_str());
    } };

    if (avs_check_version(env, 10))
        return set_error("AviSynth+ version must be r3928 or later.");
    if (avs_component_size(&fi->vi) != 4)
        return set_error("expects clip with type fp32.");

    std::unique_ptr<const AVS_VideoInfo* []> in_vis{ std::make_unique_for_overwrite< const AVS_VideoInfo * []>(num_nodes) };
    in_vis[0] = &fi->vi;
    for (int i{ 1 }; i < num_nodes; ++i)
    {
        in_vis[i] = avs_get_video_info(d->nodes[i - 1]);
        if (avs_component_size(in_vis[i]) != 4)
            return set_error("expects clip with type fp32.");
        if (in_vis[i]->width != fi->vi.width || in_vis[i]->height != fi->vi.height)
            return set_error("dimensions of clips mismatch.");
        if (in_vis[i]->num_frames != fi->vi.num_frames)
            return set_error("number of frames mismatch.");
    }

    const int device_id{ avs_defined(avs_array_elt(args, Device)) ? (avs_as_int(avs_array_elt(args, Device))) : 0 };
    d->device_id = device_id;
    const int verbosity{ avs_defined(avs_array_elt(args, Verbosity)) ? (avs_as_int(avs_array_elt(args, Verbosity))) : 3 };
    if (verbosity < 0 || verbosity > 4)
        return set_error("verbosity must be between 0..4.");

    d->overlap_w = avs_defined(avs_array_elt(args, Overlap_w)) ? (avs_as_int(avs_array_elt(args, Overlap_w))) : -4525;
    d->overlap_h = avs_defined(avs_array_elt(args, Overlap_h)) ? (avs_as_int(avs_array_elt(args, Overlap_h))) : -4525;
    if (d->overlap_w != -4525)
    {
        if (d->overlap_h == -4525)
            d->overlap_h = d->overlap_w;

        if (d->overlap_w < 0)
            return set_error("overlap_w must be non-negative.");
        if (d->overlap_h < 0)
            return set_error("overlap_h must be non-negative.");
    }
    else
    {
        d->overlap_w = 0;

        if (d->overlap_h == -4525)
            d->overlap_h = 0;

        if (d->overlap_h < 0)
            return set_error("overlap_h must be non-negative.");
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
                return set_error("tilesize_w must be specified.");

            if (d->overlap_h != 0)
                return set_error("tilesize_h must be specified.");
        }

        // set tile size to video dimensions
        tile_w = fi->vi.width;
        tile_h = fi->vi.height;
    }
    if (tile_w - 2 * d->overlap_w <= 0)
        return set_error("overlap_w too large.");
    if (tile_h - 2 * d->overlap_h <= 0)
        return set_error("overlap_h too large.");

    Backend backend{ [&]()
    {
        std::string provider{ avs_defined(avs_array_elt(args, Provider)) ? (avs_as_string(avs_array_elt(args, Provider))) : "" };
        std::transform(provider.begin(), provider.end(), provider.begin(), [](unsigned char c) { return std::tolower(c); });

        if (!provider.size() || !provider.compare("cpu"))
        {
            return Backend::CPU;
        }
        else if (!provider.compare("cuda"))
        {
            do
            {
                if (cudaError_t result = cudaSetDevice(device_id); result != cudaSuccess)
                    return Backend::UNKNOWN;
            }
            while (0);
            return Backend::CUDA;
        }
        else if (!provider.compare("dml"))
            return Backend::DML;
        else
            return Backend::UNKNOWN;
    }() };
    if (backend == Backend::UNKNOWN)
        return set_error("unknwon provider.");

#ifdef _WIN32
    if (backend == Backend::DML)
    {
        HMODULE dll_name{ GetModuleHandleW(L"onnxruntime.dll") };
        if (!dll_name)
            return set_error("no onnxruntime.dll found.");
        else
        {
            DWORD pathLen = MAX_PATH;
            std::wstring dll_path(pathLen, 0);
            DWORD result{ GetModuleFileNameW(dll_name, const_cast<wchar_t*>(dll_path.c_str()), pathLen) };
            while (result == 0 || result == pathLen)
            {
                const DWORD ret{ GetLastError() };
                if (ret == ERROR_INSUFFICIENT_BUFFER && pathLen < 32768)
                {
                    pathLen <<= 1;
                    dll_path.resize(pathLen);
                    result = GetModuleFileNameW(dll_name, const_cast<wchar_t*>(dll_path.c_str()), pathLen);
                }
                else
                    return set_error("cannot obtain onnxruntime.dll path.");
            }

            dll_path.resize(dll_path.rfind('\\') + 1);
            dll_path.append(L"DirectML.dll");

            d->dml_dll = LoadLibraryW(dll_path.c_str());
            if (!d->dml_dll)
                return set_error("failed loading DirectML.dll.");
        }
    }
#endif // _WIN32

    const int num_streams{ avs_defined(avs_array_elt(args, Num_streams)) ? (avs_as_int(avs_array_elt(args, Num_streams))) : 1 };
    if (num_streams <= 0)
        return set_error("num_streams must be positive.");
    d->num_streams = num_streams;

    const int cudnn_benchmark{ avs_defined(avs_array_elt(args, Cudnn_benchmark)) ? (avs_as_bool(avs_array_elt(args, Cudnn_benchmark))) : 1 };

    if (auto err{ ortInit() }; err.has_value())
        return set_error(err.value());

    const int fp16{ avs_defined(avs_array_elt(args, Fp16)) ? (avs_as_bool(avs_array_elt(args, Fp16))) : 0 };
    const int path_is_serialization{ avs_defined(avs_array_elt(args, Path_is_serialization)) ? (avs_as_bool(avs_array_elt(args, Path_is_serialization))) : 0 };
    const int use_cuda_graph{ avs_defined(avs_array_elt(args, Use_cuda_graph)) ? (avs_as_bool(avs_array_elt(args, Use_cuda_graph))) : 0 };
    std::string network_path{ avs_as_string(avs_array_elt(args, Network_path)) };
    if (!network_path.size())
        return set_error("network_path must be specified.");

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

    if (fp16)
    {
        std::unordered_set<std::string> fp16_blacklist_ops;
        const int num{ (avs_defined(avs_array_elt(args, Fp16_blacklist_ops))) ? avs_array_size(avs_array_elt(args, Fp16_blacklist_ops)) : 0 };
        if (num == 0)
            fp16_blacklist_ops = {
            "ArrayFeatureExtractor", "Binarizer", "CastMap", "CategoryMapper",
            "DictVectorizer", "FeatureVectorizer", "Imputer", "LabelEncoder",
            "LinearClassifier", "LinearRegressor", "Normalizer", "OneHotEncoder",
            "SVMClassifier", "SVMRegressor", "Scaler", "TreeEnsembleClassifier",
            "TreeEnsembleRegressor", "ZipMap", "NonMaxSuppression", "TopK",
            "RoiAlign", "Range", "CumSum", "Min", "Max", "Resize", "Upsample",
            "ReduceMean", // for CUGAN-pro
            "GridSample" // for RIFE, etc
        };
        else
        {
            for (int i{ 0 }; i < num; ++i)
                fp16_blacklist_ops.emplace(avs_as_string(*(avs_as_array(avs_array_elt(args, Clips)) + i)));
        }
        convert_float_to_float16(onnx_model, false, fp16_blacklist_ops);
    }

    rename(onnx_model);

    std::string onnx_data{ onnx_model.SerializeAsString() };
    if (std::size(onnx_data) == 0)
        return set_error("proto serialization failed.");

    // onnxruntime related code

    // environment per filter instance
    auto logger_id_str = "mlrt_ort" + std::to_string(logger_id.fetch_add(1, std::memory_order::relaxed));
    checkError(ortapi->CreateEnv(static_cast<OrtLoggingLevel>(verbosity), logger_id_str.c_str(), &d->environment));

    OrtMemoryInfo* memory_info;
    if (backend == Backend::CUDA)
    {
        checkError(ortapi->CreateMemoryInfo(
            "Cuda", OrtDeviceAllocator, device_id,
            OrtMemTypeDefault, &memory_info
        ));
    }
    else
    {
        checkError(ortapi->CreateMemoryInfo(
            "Cpu", OrtDeviceAllocator, /* device_id */ 0,
            OrtMemTypeDefault, &memory_info
        ));
    }

    OrtAllocator* cpu_allocator;
    checkError(ortapi->GetAllocatorWithDefaultOptions(&cpu_allocator));

    // per-stream context
    d->semaphore.current.store(num_streams - 1, std::memory_order_relaxed);
    d->tickets.reserve(num_streams);
    for (int i{ 0 }; i < num_streams; ++i)
        d->tickets.push_back(i);

    d->resources = std::make_unique_for_overwrite<Resource[]>(num_streams);
    for (int i{ 0 }; i < num_streams; ++i)
    {
        Resource resource;

        OrtSessionOptions* session_options;
        checkError(ortapi->CreateSessionOptions(&session_options));
        checkError(ortapi->SetSessionExecutionMode(
            session_options,
            ExecutionMode::ORT_SEQUENTIAL
        ));

        checkError(ortapi->DisableMemPattern(session_options));

        if (backend == Backend::CUDA)
        {
            OrtCUDAProviderOptionsV2* cuda_options;
            checkError(ortapi->CreateCUDAProviderOptions(&cuda_options));

            // should not set 'do_copy_in_default_stream' to false
            const char* keys[]{
                "device_id",
                "cudnn_conv_algo_search",
                "cudnn_conv_use_max_workspace",
                "arena_extend_strategy",
                "enable_cuda_graph"
            };
            auto device_id_str{ std::to_string(device_id) };
            const char* values[]{
                device_id_str.c_str(),
                "EXHAUSTIVE",
                "1",
                "kSameAsRequested",
                "0"
            };
            if (!cudnn_benchmark)
                values[1] = "HEURISTIC";
            if (use_cuda_graph)
            {
                values[4] = "1";
                resource.require_replay = true;
            }
            else
                resource.require_replay = false;

            checkError(ortapi->UpdateCUDAProviderOptions(cuda_options, keys, values, std::size(keys)));
            checkError(ortapi->SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_options));

            ortapi->ReleaseCUDAProviderOptions(cuda_options);
        }
        else if (backend == Backend::DML)
        {
            const OrtDmlApi* ortdmlapi{};
            checkError(ortapi->GetExecutionProviderApi("DML", ORT_API_VERSION, (const void**)&ortdmlapi));
            checkError(ortdmlapi->SessionOptionsAppendExecutionProvider_DML(session_options, device_id));
        }

        checkError(ortapi->CreateSessionFromArray(
            d->environment,
            std::data(onnx_data), std::size(onnx_data),
            session_options,
            &resource.session
        ));

        ortapi->ReleaseSessionOptions(session_options);

        if (auto err{ checkSession(resource.session) }; err.has_value())
            return set_error(err.value());

        auto input_shape{ std::get<std::array<int64_t, 4>>(
            getShape(resource.session, true)
        ) };

        if (backend == Backend::CUDA)
        {
            checkCUDAError(cudaStreamCreateWithFlags(&resource.stream, cudaStreamNonBlocking));

            resource.input.size = (
                input_shape[0] *
                input_shape[1] *
                input_shape[2] *
                input_shape[3]
                ) * sizeof(float);

            checkCUDAError(cudaMallocHost(
                &resource.input.h_data, resource.input.size,
                cudaHostAllocWriteCombined)
            );
            checkCUDAError(cudaMalloc(&resource.input.d_data, resource.input.size));

            checkError(ortapi->CreateTensorWithDataAsOrtValue(
                memory_info,
                resource.input.d_data, resource.input.size,
                std::data(input_shape), std::size(input_shape),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &resource.input_tensor
            ));
        }
        else
        {
            checkError(ortapi->CreateTensorAsOrtValue(
                cpu_allocator,
                std::data(input_shape), std::size(input_shape),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &resource.input_tensor
            ));
        }

        auto output_shape{ std::get<std::array<int64_t, 4>>(
            getShape(resource.session, false)
        ) };

        if (backend == Backend::CUDA)
        {
            resource.output.size = (
                output_shape[0] *
                output_shape[1] *
                output_shape[2] *
                output_shape[3]
                ) * sizeof(float);

            checkCUDAError(cudaMallocHost(&resource.output.h_data, resource.output.size));
            checkCUDAError(cudaMalloc(&resource.output.d_data, resource.output.size));

            checkError(ortapi->CreateTensorWithDataAsOrtValue(
                memory_info,
                resource.output.d_data, resource.output.size,
                std::data(output_shape), std::size(output_shape),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &resource.output_tensor
            ));
        }
        else
        {
            checkError(ortapi->CreateTensorAsOrtValue(
                cpu_allocator,
                std::data(output_shape), std::size(output_shape),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &resource.output_tensor
            ));
        }

        checkError(ortapi->CreateIoBinding(resource.session, &resource.binding));
        checkError(ortapi->SessionGetInputName(
            resource.session, 0, cpu_allocator, &resource.input_name
        ));

        char* output_name;
        checkError(ortapi->SessionGetOutputName(
            resource.session, 0, cpu_allocator, &resource.output_name
        ));

        checkError(ortapi->BindInput(resource.binding, resource.input_name, resource.input_tensor));
        checkError(ortapi->BindOutput(resource.binding, resource.output_name, resource.output_tensor));

        if (auto err{ checkNodesAndNetwork(resource.session, in_vis, num_nodes) }; err.has_value())
            return set_error(err.value());

        if (i == 0)
            setDimensions(&fi->vi, input_shape, output_shape);

        d->resources[i] = resource;
    }

    ortapi->ReleaseMemoryInfo(memory_info);

    AVS_Value v{ avs_new_value_clip(clip) };

    fi->user_data = reinterpret_cast<void*>(d);
    fi->set_cache_hints = set_cache_hints_mlrt_ort;

    if (backend == Backend::CPU)
    {
        fi->get_frame = get_frame_mlrt_ort<Backend::CPU>;
        fi->free_filter = free_mlrt_ort<Backend::CPU>;
    }
    else if (backend == Backend::CUDA)
    {
        fi->get_frame = get_frame_mlrt_ort<Backend::CUDA>;
        fi->free_filter = free_mlrt_ort<Backend::CUDA>;
    }
    else if (backend == Backend::DML)
    {
        fi->get_frame = get_frame_mlrt_ort<Backend::DML>;
        fi->free_filter = free_mlrt_ort<Backend::DML>;
    }

    avs_release_clip(clip);

    return v;
}

const char* AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment* env)
{
    avs_add_function(env, "mlrt_ort", "c+[network_path]s[overlap_w]i[overlap_h]i[tilesize_w]i[tilesize_h]i[provider]s[device]i[num_streams]i[verbosity]i[cudnn_benchmark]b[builtin]b[builtindir]s[fp16]b[path_is_serialization]b[use_cuda_graph]b[fp16_blacklist_ops]s*", Create_mlrt_ort, 0);
    return "mlrt_ort";
}

#include <array>
#include <optional>
#include <regex>
#include <string>
#include <variant>
#include <vector>

#include "avisynth_c.h"
#include "boost/dll/runtime_symbol_info.hpp"

#include <onnx/common/version.h>
#include <onnx/onnx_pb.h>

#include <ie/ie_core.hpp>
#include <openvino/pass/constant_folding.hpp>

#include <openvino/pass/visualize_tree.hpp>

#ifdef _WIN32
#define NOCOMM
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include <Windows.h>

static std::atomic<int> ref_count{ 0 };
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


using namespace std::string_literals;


static std::array<int, 4> getShape(const InferenceEngine::ExecutableNetwork& network, bool input)
{

    InferenceEngine::SizeVector dims;

    if (input)
        dims = network.GetInputsInfo().cbegin()->second->getTensorDesc().getDims();
    else
        dims = network.GetOutputsInfo().cbegin()->second->getTensorDesc().getDims();

    std::array<int, 4> ret;
    for (unsigned i{ 0 }; i < std::size(ret); ++i)
        ret[i] = static_cast<int>(dims[i]);

    return ret;
}


static int numPlanes(const std::vector<const AVS_VideoInfo*>& vis)
{

    int num_planes{ 0 };

    for (const auto& vi : vis)
        num_planes += avs_num_components(vi);

    return num_planes;
}


[[nodiscard]]
static std::optional<std::string> checkNodes(const std::vector<const AVS_VideoInfo*>& vis)
{
    for (const auto& vi : vis)
    {
        if (avs_component_size(vi) != 4)
            return "expects clip with type fp32";

        if (vi->width != vis[0]->width || vi->height != vis[0]->height)
            return "dimensions of clips mismatch";

        if (vi->num_frames != vis[0]->num_frames)
            return "number of frames mismatch";
    }

    return {};
}


template <typename T>
[[nodiscard]]
static std::optional<std::string> checkIOInfo(const T& info, bool is_output)
{
    if (info->getPrecision() != InferenceEngine::Precision::FP32)
        return "expects network IO with type fp32";

    const auto& desc{ info->getTensorDesc() };
    if (desc.getLayout() != InferenceEngine::Layout::NCHW)
        return "expects network IO with layout NCHW";

    const auto& dims{ desc.getDims() };
    if (dims.size() != 4)
        return "expects network with 4-D IO";

    if (dims[0] != 1)
        return "batch size of network must be 1";

    if (is_output)
    {
        auto out_channels{ dims[1] };
        if (out_channels != 1 && out_channels != 3)
            return "output dimensions must be 1 or 3";
    }

    return {};
}


[[nodiscard]]
static std::optional<std::string> checkNetwork(const InferenceEngine::CNNNetwork& network)
{
    const auto& inputs_info{ network.getInputsInfo() };

    if (auto num_inputs{ std::size(inputs_info) }; num_inputs != 1)
        return "network input count must be 1, got " + std::to_string(num_inputs);

    const auto& input_info{ inputs_info.cbegin()->second };
    if (auto err{ checkIOInfo(input_info, false) }; err.has_value())
        return err.value();

    const auto& outputs_info{ network.getOutputsInfo() };

    if (auto num_outputs{ std::size(outputs_info) }; num_outputs != 1)
        return "network output count must be 1, got " + std::to_string(num_outputs);

    const auto& output_info{ outputs_info.cbegin()->second };
    if (auto err{ checkIOInfo(output_info, true) }; err.has_value())
        return err.value();

    return {};
}


[[nodiscard]]
static std::optional<std::string> checkNodesAndNetwork(const InferenceEngine::ExecutableNetwork& network, const std::vector<const AVS_VideoInfo*>& vis)
{
    const auto& network_in_dims{ (network.GetInputsInfo().cbegin()->second->getTensorDesc().getDims()) };

    const int network_in_channels{ static_cast<int>(network_in_dims[1]) };
    const int num_planes{ numPlanes(vis) };
    if (network_in_channels != num_planes)
        return "expects " + std::to_string(network_in_channels) + " input planes";

    const int network_in_height{ static_cast<int>(network_in_dims[2]) };
    const int network_in_width{ static_cast<int>(network_in_dims[3]) };
    const int clip_in_height{ vis.front()->height };
    const int clip_in_width{ vis.front()->width };
    if (network_in_height > clip_in_height || network_in_width > clip_in_width)
        return "tile size larger than clip dimension";

    return {};
}


static void setDimensions(AVS_VideoInfo* vi, const InferenceEngine::ExecutableNetwork& network)
{
    auto in_dims{ network.GetInputsInfo().cbegin()->second->getTensorDesc().getDims() };
    auto out_dims{ network.GetOutputsInfo().cbegin()->second->getTensorDesc().getDims() };

    vi->height *= out_dims[2] / in_dims[2];
    vi->width *= out_dims[3] / in_dims[3];

    if (out_dims[1] == 1)
        vi->pixel_type = AVS_CS_Y32;
    else if (out_dims[1] == 3)
        vi->pixel_type = AVS_CS_RGBPS;
}

struct OVData
{
    std::vector<AVS_Clip*> nodes;

    int overlap_w;
    int overlap_h;

    InferenceEngine::ExecutableNetwork executable_network;
    InferenceEngine::InferRequest infer_requests;

    std::string input_name;
    std::string output_name;

    std::string err;

#ifdef _WIN32
    std::string mlrt_ov_path;
#endif // _WIN32

};


static AVS_VideoFrame* AVSC_CC get_frame_mlrt_ov(AVS_FilterInfo* fi, int n)
{
    OVData* d{ static_cast<OVData*>(fi->user_data) };

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
    auto src_bytes{ avs_component_size(in_vis.front()) };
    auto src_tile_shape{ getShape(d->executable_network, true) };
    auto src_tile_h{ src_tile_shape[2] };
    auto src_tile_w{ src_tile_shape[3] };
    auto src_tile_w_bytes{ src_tile_w * src_bytes };
    auto src_tile_bytes{ src_tile_h * src_tile_w_bytes };

    constexpr int planes_r[3] = { AVS_PLANAR_R, AVS_PLANAR_G, AVS_PLANAR_B };
    constexpr int plane_y{ AVS_PLANAR_Y };
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

    AVS_VideoFrame* dst_frame{ avs_new_video_frame_p(fi->env, &fi->vi, src_frames.front()) };

    auto dst_stride{ avs_get_pitch(dst_frame) };
    auto dst_bytes{ avs_component_size(&fi->vi) };
    auto dst_tile_shape{ getShape(d->executable_network, false) };
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
        using namespace std::string_literals;

        avs_release_video_frame(dst_frame);

        for (const auto& frame : src_frames)
            avs_release_video_frame(frame);

        d->err = "mlrt_ov: "s + error_message;
        fi->error = d->err.c_str();

        return nullptr;
    }
    };

    int y = 0;
    while (true)
    {
        const int y_crop_start{ (y == 0) ? 0 : d->overlap_h };
        const int y_crop_end{ (y == src_height - src_tile_h) ? 0 : d->overlap_h };

        int x = 0;
        while (true)
        {
            const int x_crop_start{ (x == 0) ? 0 : d->overlap_w };
            const int x_crop_end{ (x == src_width - src_tile_w) ? 0 : d->overlap_w };

            {
                InferenceEngine::Blob::Ptr input{ d->infer_requests.GetBlob(d->input_name) };

                auto minput{ input->as<InferenceEngine::MemoryBlob>() };
                auto minputHolder{ minput->wmap() };
                uint8_t* input_buffer{ minputHolder.as<uint8_t*>() };

                for (const auto& _src_ptr : src_ptrs)
                {
                    const uint8_t* src_ptr{ _src_ptr + y * src_stride + x * src_bytes };

                    avs_bit_blt(fi->env, input_buffer, src_tile_w_bytes, src_ptr, src_stride, src_tile_w_bytes, src_tile_h);

                    input_buffer += src_tile_bytes;
                }
            }

            try
            {
                d->infer_requests.Infer();
            }
            catch (const InferenceEngine::Exception& e)
            {
                return set_error("[IE exception] Create inference request: "s + e.what());
            }
            catch (const std::exception& e)
            {
                return set_error("[Standard exception] Create inference request: "s + e.what());
            }

            {
                InferenceEngine::Blob::CPtr output{ d->infer_requests.GetBlob(d->output_name) };

                auto moutput{ output->as<const InferenceEngine::MemoryBlob>() };
                auto moutputHolder{ moutput->rmap() };
                const uint8_t* output_buffer{ moutputHolder.as<const uint8_t*>() };

                for (int plane{ 0 }; plane < dst_planes; ++plane)
                {
                    uint8_t* dst_ptr{ (dst_ptrs[plane] + h_scale * y * dst_stride + w_scale * x * dst_bytes) };

                    avs_bit_blt(fi->env, dst_ptr + (y_crop_start * dst_stride + x_crop_start * dst_bytes), dst_stride, output_buffer + (y_crop_start * dst_tile_w_bytes + x_crop_start * dst_bytes), dst_tile_w_bytes,
                        dst_tile_w_bytes - (x_crop_start + x_crop_end) * dst_bytes, dst_tile_h - (y_crop_start + y_crop_end));

                    output_buffer += dst_tile_bytes;
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

    for (const auto& frame : src_frames)
        avs_release_video_frame(frame);

    return dst_frame;
}


static void AVSC_CC free_mlrt_ov(AVS_FilterInfo* fi)
{
    OVData* d{ static_cast<OVData*>(fi->user_data) };

    for (const auto& node : d->nodes)
        avs_release_clip(node);

#ifdef _WIN32
    if (--ref_count == 0)
    {
        std::array<HMODULE, 2> loaded_dlls{
            GetModuleHandleA((d->mlrt_ov_path + "/mlrt_ov_rt/openvino.dll").c_str()),
            GetModuleHandleA((d->mlrt_ov_path + "/mlrt_ov_rt/tbb12.dll").c_str())
        };

        for (int i{ 0 }; i < 2; ++i)
            FreeLibrary(loaded_dlls[i]);
    }
#endif // _WIN32

    delete d;
}


static int AVSC_CC set_cache_hints_mlrt_ov(AVS_FilterInfo* fi, int cachehints, int frame_range)
{
    return cachehints == AVS_CACHE_GET_MTMODE ? 2 : 0;
}


static AVS_Value AVSC_CC Create_mlrt_ov(AVS_ScriptEnvironment* env, AVS_Value args, void* param)
{
    enum { Clips, Network_path, Overlap_w, Overlap_h, Tilesize_w, Tilesize_h, Device, Builtin, Builtindir, Fp16, Config, Path_is_serialization, List_devices, Fp16_blacklist_ops, Dot_path };

    const std::string mlrt_ov_path{ boost::dll::this_line_location().parent_path().generic_string() };

#ifdef _WIN32
    if (ref_count == 0)
    {
        std::array<HMODULE, 2> loaded_dlls{
            LoadLibraryA((mlrt_ov_path + "/mlrt_ov_rt/tbb12.dll").c_str()),
            LoadLibraryA((mlrt_ov_path + "/mlrt_ov_rt/openvino.dll").c_str()),
        };
        if (!loaded_dlls[0])
        {
            std::string_view tbb12_path{ ("mlrt_ov: cannot find " + mlrt_ov_path + "/mlrt_ov_rt/tbb12.dll").c_str() };
            return avs_new_value_error(avs_save_string(env, tbb12_path.data(), tbb12_path.size()));
        }
        if (!loaded_dlls[1])
        {
            FreeLibrary(loaded_dlls[0]);
            std::string_view openvino_path{ ("mlrt_ov: cannot find " + mlrt_ov_path + "/mlrt_ov_rt/openvino.dll").c_str() };
            return avs_new_value_error(avs_save_string(env, openvino_path.data(), openvino_path.size()));
        }
    }

    ++ref_count;
#endif // _WIN32

    OVData* d{ new OVData() };

#ifdef _WIN32
    d->mlrt_ov_path = mlrt_ov_path;
#endif // _WIN32

    AVS_FilterInfo* fi;
    AVS_Clip* clip{ avs_new_c_filter(env, &fi, *avs_as_array(avs_array_elt(args, Clips)), 1) };

    if (avs_defined(avs_array_elt(args, List_devices)) ? avs_as_bool(avs_array_elt(args, List_devices)) : 0)
    {
        try
        {
            auto core{ InferenceEngine::Core() };
            auto devices{ core.GetAvailableDevices() };
            for (const auto& device : devices)
                d->err += device;

            AVS_Value cl{ avs_new_value_clip(clip) };
            AVS_Value args_[2]{ cl, avs_new_value_string(d->err.c_str()) };
            AVS_Value inv{ avs_invoke(fi->env, "Text", avs_new_value_array(args_, 2), 0) };

            avs_release_value(cl);
            avs_release_clip(clip);

            return inv;
        }
        catch (const InferenceEngine::Exception& e)
        {
            d->err = "[IE exception] Initialize inference engine: "s + e.what();
            avs_release_clip(clip);
            return avs_new_value_error(d->err.c_str());
        }
        catch (const std::exception& e)
        {
            d->err = "[Standard exception] Initialize inference engine: "s + e.what();
            avs_release_clip(clip);
            return avs_new_value_error(d->err.c_str());
        }
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

        d->err = "mlrt_ov: "s + error_message;

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

    const char* device{ avs_defined(avs_array_elt(args, Device)) ? (avs_as_string(avs_array_elt(args, Device))) : "CPU" };

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

    const bool fp16{ avs_defined(avs_array_elt(args, Fp16)) ? static_cast<const bool>(!!avs_as_bool(avs_array_elt(args, Fp16))) : false };

    const bool path_is_serialization{ static_cast<bool>(!!avs_defined(avs_array_elt(args, Path_is_serialization)) ? (avs_as_bool(avs_array_elt(args, Path_is_serialization))) : false) };
    std::string network_path{ avs_as_string(avs_array_elt(args, Network_path)) };
    if (!network_path.size())
        return set_error("network_path must be specified");

    const bool builtin{ static_cast<bool>(!!avs_defined(avs_array_elt(args, Builtin)) ? (avs_as_bool(avs_array_elt(args, Builtin))) : true) };
    if (builtin)
    {
        std::string modeldir{ avs_defined(avs_array_elt(args, Builtindir)) ? (avs_as_string(avs_array_elt(args, Builtindir))) : "models" };
        network_path = mlrt_ov_path + "/" + modeldir + "/" + network_path;
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
            "RoiAlign", "Range", "CumSum", "Min", "Max"
        };
        else
        {
            for (int i{ 0 }; i < num; ++i)
                fp16_blacklist_ops.emplace(avs_as_string(*(avs_as_array(avs_array_elt(args, Clips)) + i)));
        }
        convert_float_to_float16(onnx_model, false, fp16_blacklist_ops);
    }

    std::string onnx_data{ onnx_model.SerializeAsString() };
    if (std::size(onnx_data) == 0)
        return set_error("proto serialization failed");

    {
        InferenceEngine::Core core;
        InferenceEngine::CNNNetwork network;
        try
        {
            auto empty = InferenceEngine::Blob::CPtr();
            network = core.ReadNetwork(onnx_data, empty);
        }
        catch (const InferenceEngine::Exception& e)
        {
            return set_error("[IE exception] ReadNetwork(): "s + e.what());
        }
        catch (const std::exception& e)
        {
            return set_error("[Standard exception] ReadNetwork(): "s + e.what());
        }

        if (auto err = checkNetwork(network); err.has_value())
            return set_error(err.value());

        auto function = network.getFunction(); // mutable

        try
        {
            ov::pass::ConstantFolding().run_on_model(function);
        }
        catch (const ov::Exception& e)
        {
            return set_error(e.what());
        }

        if (avs_defined(avs_array_elt(args, Dot_path)))
        {
            try
            {
                ov::pass::VisualizeTree(avs_as_string(avs_array_elt(args, Dot_path)), nullptr, true).run_on_model(function);
            }
            catch (const ov::Exception& e)
            {
                return set_error(e.what());
            }
        }

        std::map<std::string, std::string> config;

        if (avs_defined(avs_array_elt(args, Config)))
        {
            std::string config_param{ avs_as_string(avs_array_elt(args, Config)) };

            int num_spaces{ 0 };
            int num_equals{ -1 };
            for (auto& string : config_param)
            {
                if (string == ' ')
                    ++num_spaces;
                if (string == '=')
                    ++num_equals;
            }
            if (num_spaces != num_equals)
                return set_error("failed parsing config.");

            std::string reg_parse{ "(\\w+)=([^ >]+)" };
            for (int i{ 0 }; i < num_spaces; ++i)
                reg_parse += "(?: (\\w+)=([^ >]+))";

            std::regex reg(reg_parse);
            std::smatch match;
            if (!std::regex_match(config_param.cbegin(), config_param.cend(), match, reg))
                return set_error("failed parsing config.");

            for (int i = 1; match[i + 1].matched; i += 2)
                config[match[i].str()] = match[i + 1].str();
        }

        try
        {
            d->executable_network = core.LoadNetwork(network, device, config);
        }
        catch (const InferenceEngine::Exception& e)
        {
            return set_error(e.what());
        }

        if (auto err = checkNodesAndNetwork(d->executable_network, in_vis); err.has_value())
            return set_error(err.value());

        setDimensions(&fi->vi, d->executable_network);

        d->input_name = d->executable_network.GetInputsInfo().cbegin()->first;
        d->output_name = d->executable_network.GetOutputsInfo().cbegin()->first;

        try
        {
            d->infer_requests = d->executable_network.CreateInferRequest();
        }
        catch (const InferenceEngine::Exception& e)
        {
            return set_error("[IE exception] Create inference request: "s + e.what());
        }
        catch (const std::exception& e)
        {
            return set_error("[Standard exception] Create inference request: "s + e.what());
        }

        AVS_Value v{ avs_new_value_clip(clip) };

        fi->user_data = reinterpret_cast<void*>(d);
        fi->get_frame = get_frame_mlrt_ov;
        fi->set_cache_hints = set_cache_hints_mlrt_ov;
        fi->free_filter = free_mlrt_ov;

        avs_release_clip(clip);

        return v;
    }
}

const char* AVSC_CC avisynth_c_plugin_init(AVS_ScriptEnvironment* env)
{
    const char args[]{
        "c+"
        "[network_path]s"
        "[overlap_w]i"
        "[overlap_h]i"
        "[tilesize_w]i"
        "[tilesize_h]i"
        "[device]s"
        "[builtin]b"
        "[builtindir]s"
        "[fp16]b"
        "[config]s"
        "[path_is_serialization]b"
        "[list_devices]b"
        "[fp16_blacklist_ops]s*"
        "[dot_path]s"
    };

    avs_add_function(env, "mlrt_ov", args, Create_mlrt_ov, 0);

    return "mlrt_ov";
}

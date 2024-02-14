#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

#include "onnx/onnx_pb.h"
#include "onnx/shape_inference/implementation.h"


using namespace std::string_literals;

#ifdef _WIN32
#include <Windows.h>
static inline std::wstring translateName(const std::string& name, const unsigned CP) noexcept
{
    const int num_chars{ MultiByteToWideChar(CP, 0, &name[0], static_cast<int>(name.size()), nullptr, 0) };
    std::wstring wide_source(num_chars, 0);
    MultiByteToWideChar(CP, 0, &name[0], static_cast<int>(name.size()), &wide_source[0], num_chars);
    return wide_source;
}
#else
#define translateName(n, m) (n)
#endif


[[nodiscard]]
static std::optional<std::string> specifyShape(
    ONNX_NAMESPACE::ModelProto& model,
    int64_t tile_w,
    int64_t tile_h,
    int64_t batch = 1
) noexcept {

    if (model.graph().input_size() != 1)
    {
        return "graph must has a single input";
    }
    ONNX_NAMESPACE::TensorShapeProto* input_shape{
        model
            .mutable_graph()
            ->mutable_input(0)
            ->mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape()
    };

    if (model.graph().output_size() != 1)
    {
        return "graph must has a single output";
    }
    ONNX_NAMESPACE::TensorShapeProto* output_shape{
        model
            .mutable_graph()
            ->mutable_output(0)
            ->mutable_type()
            ->mutable_tensor_type()
            ->mutable_shape()
    };

    constexpr auto n_idx{ 0 };
    constexpr auto h_idx{ 2 };
    constexpr auto w_idx{ 3 };

    if (input_shape->dim_size() != 4)
    {
        return "input dimension must be 4";
    }

    input_shape->mutable_dim(n_idx)->set_dim_value(batch);
    input_shape->mutable_dim(h_idx)->set_dim_value(tile_h);
    input_shape->mutable_dim(w_idx)->set_dim_value(tile_w);

    if (output_shape->dim_size() != 4)
    {
        return "output dimsion must be 4";
    }

    output_shape->mutable_dim(n_idx)->set_dim_value(batch);
    output_shape->mutable_dim(h_idx)->clear_dim_value();
    output_shape->mutable_dim(w_idx)->clear_dim_value();

    // remove shape info
    if (model.graph().value_info_size() != 0)
    {
        model.mutable_graph()->mutable_value_info()->Clear();
    }

    try {
        ONNX_NAMESPACE::shape_inference::InferShapes(model);
    }
    catch (const ONNX_NAMESPACE::InferenceError& e)
    {
        return e.what();
    }

    return {};
}


std::variant<std::string, ONNX_NAMESPACE::ModelProto> loadONNX(
    const std::string& path,
    const unsigned CP,
    int64_t tile_w,
    int64_t tile_h,
    bool path_is_serialization
) noexcept {

    ONNX_NAMESPACE::ModelProto onnx_proto;

    if (path_is_serialization)
    {
        if (!onnx_proto.ParseFromArray(path.data(), static_cast<int>(path.size())))
            return "parse onnx serialization failed"s;
    }
    else
    {
        std::ifstream onnx_stream(
            translateName(path, CP),
            std::ios::binary
        );

        if (!onnx_stream.good())
            return "open "s + path + " failed to open"s;
        if (!onnx_proto.ParseFromIstream(&onnx_stream))
            return "parse "s + path + " failed to parse"s;
    }

    if (auto err{ specifyShape(onnx_proto, tile_w, tile_h) }; err.has_value())
        return err.value();

    return onnx_proto;
}

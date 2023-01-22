#pragma once

#include <optional>
#include <string>
#include <tuple>

#include <onnx/onnx_pb.h>

extern std::optional<std::tuple<char*, unsigned char*>> onnx2ncnn(ONNX_NAMESPACE::ModelProto& model);

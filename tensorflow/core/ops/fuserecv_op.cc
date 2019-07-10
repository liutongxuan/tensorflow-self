#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("_FuseRecv")
    .Output("tensors: Tout")
    .Attr("Tout: list(type)")
    .Attr("tensor_types: list(type)")
    .Attr("tensor_names: list(string)")
    .Attr("send_devices: list(string)")
    .Attr("recv_devices: list(string)")
    .Attr("send_device_incarnations: list(int)")
    .Attr("client_terminated: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(placeholder)doc");

REGISTER_OP("_HostFuseRecv")
    .Output("tensors: Tout")
    .Attr("Tout: list(type)")
    .Attr("tensor_types: list(type)")
    .Attr("tensor_names: list(string)")
    .Attr("send_devices: list(string)")
    .Attr("recv_devices: list(string)")
    .Attr("send_device_incarnations: list(int)")
    .Attr("client_terminated: bool = false")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(placeholder)doc");
}
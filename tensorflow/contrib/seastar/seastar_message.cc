#include "tensorflow/contrib/seastar/seastar_message.h"

namespace tensorflow {

void SeastarMessage::DeserializeMessage(SeastarMessage* sm,
                                        const char* message) {
  memcpy(&sm->is_dead_, &message[kIsDeadStartIndex], sizeof(sm->is_dead_));
  memcpy(&sm->send_start_micros_, &message[kSendMicrosStartIndex],
         sizeof(sm->send_start_micros_));
  memcpy(&sm->data_type_, &message[kDataTypeStartIndex],
         sizeof(sm->data_type_));
  memcpy(&sm->tensor_shape_, &message[kTensorShapeStartIndex],
         sizeof(sm->tensor_shape_));
  memcpy(&sm->tensor_bytes_, &message[kTensorBytesStartIndex],
         sizeof(sm->tensor_bytes_));
}

void SeastarMessage::SerializeMessage(const SeastarMessage& sm, char* message) {
  memcpy(&message[kIsDeadStartIndex], &sm.is_dead_, sizeof(sm.is_dead_));
  memcpy(&message[kSendMicrosStartIndex], &sm.send_start_micros_,
         sizeof(sm.send_start_micros_));
  memcpy(&message[kDataTypeStartIndex], &sm.data_type_, sizeof(sm.data_type_));
  memcpy(&message[kTensorShapeStartIndex], &sm.tensor_shape_,
         sizeof(sm.tensor_shape_));
  memcpy(&message[kTensorBytesStartIndex], &sm.tensor_bytes_,
         sizeof(sm.tensor_bytes_));
}

uint64_t SeastarMessage::SerializeTensorMessage(const Tensor& in,
                                                const TensorProto& inp,
                                                bool is_dead,
                                                uint64_t send_start_micros,
                                                SeastarBuf* message_buf,
                                                SeastarBuf* tensor_buf) {
  SeastarMessage sm;
  sm.tensor_shape_ = in.shape();
  sm.data_type_ = in.dtype();
  sm.is_dead_ = is_dead;
  sm.send_start_micros_ = send_start_micros;

  bool can_memcpy = DataTypeCanUseMemcpy(sm.data_type_);

  if (can_memcpy) {
    sm.tensor_bytes_ = in.TotalBytes();
    tensor_buf->len_ = sm.tensor_bytes_;
    tensor_buf->data_ = const_cast<char*>(in.tensor_data().data());
    tensor_buf->owned_ = false;
  } else {
    sm.tensor_bytes_ = inp.ByteSize();
    tensor_buf->len_ = sm.tensor_bytes_;
    tensor_buf->data_ = new char[tensor_buf->len_]();
    inp.SerializeToArray(tensor_buf->data_, tensor_buf->len_);
  }

  message_buf->len_ = SeastarMessage::kMessageTotalBytes;
  message_buf->data_ = new char[message_buf->len_];
  SeastarMessage::SerializeMessage(sm, message_buf->data_);

  return SeastarMessage::kMessageTotalBytes + sm.tensor_bytes_;
}

}  // namespace tensorflow

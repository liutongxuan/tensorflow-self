#include "tensorflow/contrib/seastar/seastar_tag_factory.h"

#include "tensorflow/contrib/seastar/seastar_message.h"
#include "tensorflow/contrib/seastar/seastar_tensor_coding.h"

namespace tensorflow {

SeastarTagFactory::SeastarTagFactory(SeastarWorkerService* worker_service)
    : worker_service_(worker_service) {}

SeastarClientTag* SeastarTagFactory::CreateSeastarClientTag(
    seastar::temporary_buffer<char>& header) {
  char* p = const_cast<char*>(header.get());

  // Igonre the BBBB and method segment.
  SeastarClientTag* tag = NULL;
  memcpy(&tag, p + SeastarClientTag::kTagIndex, 8);
  memcpy(&tag->status_, p + SeastarClientTag::kStatusIndex, 4);

  if (tag->status_ != 0) {
    memcpy(&tag->err_msg_len_, p + SeastarClientTag::kPayloadLenIndex, 8);
    return tag;
  }

  if (!tag->IsRecvTensor()) {
    memcpy(&tag->resp_body_buf_.len_, p + SeastarClientTag::kPayloadLenIndex, 8);
    tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_];
  }
  return tag;
}

SeastarServerTag* SeastarTagFactory::CreateSeastarServerTag(
    seastar::temporary_buffer<char>& header,
    seastar::channel* seastar_channel) {
  char* p = const_cast<char*>(header.get());
  SeastarServerTag* tag = new SeastarServerTag(seastar_channel,
                                               worker_service_);
  // Ignore the BBBB segment.
  memcpy(&tag->method_, p + SeastarClientTag::kMethodIndex, 4);
  memcpy(&tag->client_tag_id_, p + SeastarClientTag::kTagIndex, 8);
  // Igonre the status segment

  memcpy(&(tag->req_body_buf_.len_), p + SeastarClientTag::kPayloadLenIndex, 8);
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_];

  return tag;
}

}  // namespace tensorflow

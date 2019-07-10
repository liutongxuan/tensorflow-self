#include "tensorflow/contrib/seastar/seastar_server_tag.h"

#include "tensorflow/contrib/seastar/seastar_message.h"
#include "tensorflow/contrib/seastar/seastar_worker_service.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
void InitStarServerTag(protobuf::Message* request,
                       protobuf::Message* response,
                       StarServerTag* tag) {
  request->ParseFromArray(tag->req_body_buf_.data_,
                          tag->req_body_buf_.len_);

  StatusCallback done = [response, tag] (const Status& s) {
    tag->resp_header_buf_.len_ = StarServerTag::kHeaderSize;
    tag->resp_header_buf_.data_ = new char[tag->resp_header_buf_.len_]();

    memcpy(tag->resp_header_buf_.data_, "BBBB", 4);
    // Ingore method segment.
    memcpy(tag->resp_header_buf_.data_ + StarServerTag::kTagIndex,
           &tag->client_tag_id_, 8);
    tag->status_ = static_cast<int32>(s.code());
    memcpy(tag->resp_header_buf_.data_ + StarServerTag::kStatusIndex,
           &tag->status_, 4);
    // Ingore user data segment.

    if (s.ok()) {
      tag->resp_body_buf_.len_ = response->ByteSize();
      tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_]();
      response->SerializeToArray(tag->resp_body_buf_.data_, tag->resp_body_buf_.len_);
      memcpy(tag->resp_header_buf_.data_ + StarServerTag::kPayloadLenIndex,
             &tag->resp_body_buf_.len_, 8);

    } else {
      //TODO: RemoteWorker::LoggingRequest doesn't need to response.
      //      can be more elegant.
      // Send err msg back to client

      tag->resp_body_buf_.len_ = s.error_message().length();
      tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_]();
      memcpy(tag->resp_body_buf_.data_, s.error_message().c_str(),
             tag->resp_body_buf_.len_);
      memcpy(tag->resp_header_buf_.data_ + StarServerTag::kPayloadLenIndex,
             &tag->resp_body_buf_.len_, 8);
    }

    tag->StartResp();
  };

  tag->send_resp_ = std::move(done);
  tag->clear_ = [](const Status& s) {};
}

void InitStarServerTag(protobuf::Message* request,
                       StarTensorResponse* response,
                       StarServerTag* tag,
                       StatusCallback clear) {
  request->ParseFromArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  StatusCallback done = [request, response, tag] (const Status& s) {
    tag->resp_header_buf_.len_ = StarServerTag::kHeaderSize;
    tag->resp_header_buf_.data_ = new char[tag->resp_header_buf_.len_]();

    memcpy(tag->resp_header_buf_.data_, "BBBB", 4);
    // Ingore method segment.
    memcpy(tag->resp_header_buf_.data_ + StarServerTag::kTagIndex,
           &tag->client_tag_id_, 8);
    tag->status_ = static_cast<int32>(s.code());
    memcpy(tag->resp_header_buf_.data_ + StarServerTag::kStatusIndex,
           &tag->status_, 4);
    // Ingore user data segment.

    if (s.ok()) {
      tag->InitResponseTensorBufs(1);
      uint64_t payload_len
        = StarMessage::SerializeTensorMessage(response->GetTensor(),
                                              response->GetTensorProto(),
                                              response->GetIsDead(),
                                              &tag->resp_message_bufs_[0],
                                              &tag->resp_tensor_bufs_[0]);
      memcpy(tag->resp_header_buf_.data_ + StarServerTag::kPayloadLenIndex,
             &payload_len, 8);

    } else {
      tag->resp_body_buf_.len_ = s.error_message().length();
      tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_]();
      memcpy(tag->resp_body_buf_.data_, s.error_message().c_str(),
             tag->resp_body_buf_.len_);
      memcpy(tag->resp_header_buf_.data_ + StarServerTag::kPayloadLenIndex,
             &tag->resp_body_buf_.len_, 8);
    }

    tag->StartResp();
  };

  // used for zero copy sending tensor
  tag->send_resp_ = std::move(done);
  tag->clear_ = std::move(clear);
}

void InitStarServerTag(protobuf::Message* request,
                       StarFuseTensorResponse* response,
                       StarServerTag* tag,
                       StatusCallback clear) {
  request->ParseFromArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  StatusCallback done = [request, response, tag] (const Status& s) {
    tag->resp_header_buf_.len_ = StarServerTag::kHeaderSize;
    tag->resp_header_buf_.data_ = new char[tag->resp_header_buf_.len_]();

    memcpy(tag->resp_header_buf_.data_, "BBBB", 4);
    // Ingore method segment.
    memcpy(tag->resp_header_buf_.data_ + StarServerTag::kTagIndex,
           &tag->client_tag_id_, 8);
    tag->status_ = static_cast<int32>(s.code());
    memcpy(tag->resp_header_buf_.data_ + StarServerTag::kStatusIndex,
           &tag->status_, 4);
    // Ingore user data segment.

    if (s.ok()) {
      tag->InitResponseTensorBufs(response->GetFuseCount());
      uint64_t payload_len = 0;
      for (int idx = 0; idx < tag->resp_tensor_count_; ++idx) {
        payload_len
          += StarMessage::SerializeTensorMessage(response->GetTensorByIndex(idx),
                                                 response->GetTensorProtoByIndex(idx),
                                                 response->GetIsDeadByIndex(idx),
                                                 &tag->resp_message_bufs_[idx],
                                                 &tag->resp_tensor_bufs_[idx]);
      }
      memcpy(tag->resp_header_buf_.data_ + StarServerTag::kPayloadLenIndex,
             &payload_len, 8);
    } else {
      tag->resp_body_buf_.len_ = s.error_message().length();
      tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_]();
      memcpy(tag->resp_body_buf_.data_, s.error_message().c_str(),
             tag->resp_body_buf_.len_);
      memcpy(tag->resp_header_buf_.data_ + StarServerTag::kPayloadLenIndex,
             &tag->resp_body_buf_.len_, 8);
    }

    tag->StartResp();
  };
  // used for zero copy sending tensor, unref Tensor object after star send done
  tag->clear_ = std::move(clear);
  tag->send_resp_ = std::move(done);
}

void SeastarServerTag::StartResp() {
  if (IsRecvTensor()) {
    seastar_channel_->put(ToUserPacketWithTensors());
  } else {
    seastar_channel_->put(ToUserPacket());
  }
}

seastar::user_packet* SeastarServerTag::ToUserPacket() {
  seastar::net::fragment respHeader {resp_header_buf_.data_, resp_header_buf_.len_};
  seastar::net::fragment respBody {resp_body_buf_.data_, resp_body_buf_.len_};

  std::vector<seastar::net::fragment> frags = { respHeader, respBody };
  seastar::user_packet* up = new seastar::user_packet;
  up->_fragments = frags;
  up->_done = [this](){ this->SendRespDone(); };

  return up;
}

std::vector<seastar::user_packet*> SeastarServerTag::ToUserPacketWithTensors() {
  int64 total_len = 0;
  std::vector<seastar::user_packet*> ret;
  auto up = new seastar::user_packet;

  std::vector<seastar::net::fragment> frags;
  frags.emplace_back(seastar::net::fragment {resp_header_buf_.data_,
      resp_header_buf_.len_});
  total_len += resp_header_buf_.len_;

  if (IsStarRunGraph() || status_ != 0) {
    frags.emplace_back(seastar::net::fragment {resp_body_buf_.data_,
          resp_body_buf_.len_});
    total_len += resp_body_buf_.len_;
  }

  // For fuse recv / zero copy run graph, if error happens 'resp_tensor_count_'
  // is zero as when it is inited, so no tensor can be sent.
  for (auto i = 0; i < resp_tensor_count_; ++i) {
    if (frags.size() > IOV_MAX / 2) {
      std::swap(up->_fragments, frags);

      up->_done = []() {};
      ret.emplace_back(up);
      up = new seastar::user_packet;
    }
    frags.emplace_back(seastar::net::fragment {resp_message_bufs_[i].data_,
        resp_message_bufs_[i].len_});
    total_len += resp_message_bufs_[i].len_;

    if (resp_tensor_bufs_[i].len_ > 0) {
      frags.emplace_back(seastar::net::fragment {resp_tensor_bufs_[i].data_,
        resp_tensor_bufs_[i].len_});
      total_len += resp_tensor_bufs_[i].len_;
    }
  }
  up->_fragments = frags;
  up->_done = [this]() { this->SendRespDone(); };
  ret.emplace_back(up);

  return ret;
}

Status SeastarServerTag::ParseTensor() {
  return parse_tensor_();
}

Status SeastarServerTag::ParseMessage(int idx, const char* tensor_msg, size_t len) {
  return parse_message_(idx, tensor_msg, len);
}

Status SeastarServerTag::ParseMetaData(const char* buf, size_t len) {
  return parse_meta_data_(buf, len);
}

int SeastarServerTag::GetReqTensorCount() {
  return req_tensor_count_;
}

void SeastarServerTag::InitRequestTensorBufs(int count) {
  req_tensor_count_ = count;
  req_tensor_bufs_.resize(count);
}

uint64_t SeastarServerTag::GetRequestTensorSize(int idx) {
  return req_tensor_bufs_[idx].len_;
}

char* SeastarServerTag::GetRequestTensorBuffer(int idx) {
  return req_tensor_bufs_[idx].data_;
}

SeastarServerTag::SeastarServerTag(SeastarWorkerService* seastar_worker_service)
  : method_(SeastarWorkerServiceMethod::kInvalid),
    client_tag_id_(0),
    status_(0),
    req_tensor_count_(0),
    resp_tensor_count_(0),
    parse_meta_data_(nullptr),
    parse_message_(nullptr),
    parse_tensor_(nullptr),
    send_resp_(nullptr),
    clear_(nullptr),
    seastar_worker_service_(seastar_worker_service) {
}

SeastarServerTag::~SeastarServerTag() {
  delete [] req_body_buf_.data_;
  delete [] resp_header_buf_.data_;
  delete [] resp_body_buf_.data_;

  for (int i = 0; i < resp_tensor_count_; ++i) {
    delete [] resp_message_bufs_[i].data_;
    if (resp_tensor_bufs_[i].owned_) {
      delete [] resp_tensor_bufs_[i].data_;
    }
  }

  for (uint64_t i = 0; i < req_tensor_count_; ++i) {
    if (req_tensor_bufs_[i].owned_) {
      delete [] req_tensor_bufs_[i].data_;
    }
  }
}

void SeastarServerTag::InitResponseTensorBufs(int32_t resp_tensor_count) {
  resp_tensor_count_ = resp_tensor_count;
  resp_message_bufs_.resize(resp_tensor_count);
  resp_tensor_bufs_.resize(resp_tensor_count);
}

bool SeastarServerTag::IsRecvTensor() {
  return method_ == SeastarWorkerServiceMethod::kRecvTensor
    || method_ == SeastarWorkerServiceMethod::kFuseRecvTensor;
}

void SeastarServerTag::RecvReqDone(Status s) {
  if (!s.ok()) {
    this->send_resp_(s);
    return;
  }

  auto handle = seastar_worker_service_->GetHandler(method_);
  (seastar_worker_service_->*handle)(this);
}

void SeastarServerTag::SendRespDone() {
  clear_(Status());
  delete this;
}

// called when request has been processed, mainly serialize resp to wire-format,
// and send response
void SeastarServerTag::ProcessDone(Status s) {
  send_resp_(s);
}

uint64_t SeastarServerTag::GetRequestBodySize() {
  return req_body_buf_.len_;
}

char* SeastarServerTag::GetRequestBodyBuffer() {
  return req_body_buf_.data_;
}

}  // namespace tensorflow

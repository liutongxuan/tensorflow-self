#include "tensorflow/contrib/seastar/seastar_server_tag.h"

#include "tensorflow/contrib/seastar/seastar_message.h"
#include "tensorflow/contrib/seastar/seastar_worker_service.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/lib/monitoring/cat_reporter.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {

uint64_t SerializeTensorMessage(SeastarTensorResponse* response,
    SeastarBuf* message_buf,SeastarBuf* tensor_buf) {
  SeastarMessage sm;
  Tensor& in = response->GetTensor();
  TensorProto& inp = response->GetTensorProto();

  sm.tensor_shape_ = in.shape();
  sm.data_type_ = in.dtype();
  sm.is_dead_ = response->GetIsDead();
  sm.send_start_micros_ = response->send_start_micros();

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

}  // namespace

void InitSeastarServerTag(protobuf::Message* request,
                          protobuf::Message* response, SeastarServerTag* tag) {
  request->ParseFromArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  StatusCallback done = [response, tag](const Status& s) {
    tag->resp_header_buf_.len_ = SeastarServerTag::HEADER_SIZE;
    if (!s.ok()) {
      tag->resp_header_buf_.len_ += s.error_message().length();
    }
    tag->resp_header_buf_.data_ = new char[tag->resp_header_buf_.len_]();

    memcpy(tag->resp_header_buf_.data_, SeastarServerTag::HEADER_SIGN, 8);
    memcpy(tag->resp_header_buf_.data_ + 8, &tag->client_tag_id_, 8);
    memcpy(tag->resp_header_buf_.data_ + 16, &tag->method_, 4);
    auto code = static_cast<int16_t>(s.code());
    memcpy(tag->resp_header_buf_.data_ + 20, &code, 2);

    if (s.ok()) {
      uint16_t err_len = 0;
      memcpy(tag->resp_header_buf_.data_ + 22, &err_len, 2);
      tag->resp_body_buf_.len_ = response->ByteSize();
      tag->resp_body_buf_.data_ = new char[tag->resp_body_buf_.len_]();
      response->SerializeToArray(tag->resp_body_buf_.data_,
                                 tag->resp_body_buf_.len_);

      memcpy(tag->resp_header_buf_.data_ + 24, &tag->resp_body_buf_.len_, 8);
    } else {
      // TODO: RemoteWorker::LoggingRequest doesn't need to response.
      //      can be more elegant.
      uint16_t err_len =
          std::min(UINT16_MAX, (int) (s.error_message().length()));
      memcpy(tag->resp_header_buf_.data_ + 22, &err_len, 2);
      memcpy(tag->resp_header_buf_.data_ + 32, s.error_message().c_str(),
             err_len);
    }

    tag->StartResp();
  };

  tag->send_resp_ = std::move(done);
  tag->clear_ = [](const Status& s) {};
}

void InitSeastarServerTag(protobuf::Message* request,
                          SeastarTensorResponse* response,
                          SeastarServerTag* tag, StatusCallback clear) {
  request->ParseFromArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  StatusCallback done = [request, response, tag](const Status& s) {
    tag->resp_header_buf_.len_ = SeastarServerTag::HEADER_SIZE;
    if (!s.ok()) {
      tag->resp_header_buf_.len_ += s.error_message().length();
    }
    tag->resp_header_buf_.data_ = new char[tag->resp_header_buf_.len_]();

    memcpy(tag->resp_header_buf_.data_, SeastarServerTag::HEADER_SIGN, 8);
    memcpy(tag->resp_header_buf_.data_ + 8, &tag->client_tag_id_, 8);
    memcpy(tag->resp_header_buf_.data_ + 16, &tag->method_, 4);
    auto code = static_cast<int16_t>(s.code());
    memcpy(tag->resp_header_buf_.data_ + 20, &code, 2);

    if (s.ok()) {
      uint16_t err_len = 0;
      memcpy(tag->resp_header_buf_.data_ + 22, &err_len, 2);
      SerializeTensorMessage(response, &tag->resp_message_buf_,
                             &tag->resp_tensor_buf_);
    } else {
      uint16_t err_len =
          std::min(UINT16_MAX, (int) (s.error_message().length()));
      memcpy(tag->resp_header_buf_.data_ + 22, &err_len, 2);
      memcpy(tag->resp_header_buf_.data_ + 32, s.error_message().c_str(),
             err_len);
    }

    tag->StartRespWithTensor();
  };

  // used for zero copy sending tensor
  tag->clear_ = std::move(clear);
  tag->send_resp_ = std::move(done);
}

void InitSeastarServerTag(protobuf::Message *request,
                          SeastarFuseTensorResponse *response,
                          SeastarServerTag *tag,
                          StatusCallback clear) {
  request->ParseFromArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  StatusCallback done = [request, response, tag](const Status &s) {
    tag->resp_header_buf_.len_ = SeastarServerTag::HEADER_SIZE;
    if (!s.ok()) {
      tag->resp_header_buf_.len_ += s.error_message().length();
    }
    tag->resp_header_buf_.data_ = new char[tag->resp_header_buf_.len_]();

    memcpy(tag->resp_header_buf_.data_, SeastarServerTag::HEADER_SIGN, 8);
    memcpy(tag->resp_header_buf_.data_ + 8, &tag->client_tag_id_, 8);
    memcpy(tag->resp_header_buf_.data_ + 16, &tag->method_, 4);
    auto code = static_cast<int16_t>(s.code());
    memcpy(tag->resp_header_buf_.data_ + 20, &code, 2);

    if (s.ok()) {
      uint16_t err_len = 0;
      memcpy(tag->resp_header_buf_.data_ + 22, &err_len, 2);

      tag->InitResponseTensorBufs(response->GetFuseCount());
      uint64_t send_time_micros = response->send_start_micros();
      for (int idx = 0; idx < tag->resp_tensor_count_; ++idx) {
        SeastarMessage::SerializeTensorMessage(response->GetTensorByIndex(idx),
                                                   response->GetTensorProtoByIndex(idx),
                                                   response->GetIsDeadByIndex(idx),
                                                   (idx == 0)?send_time_micros:0,
                                                   &tag->resp_message_bufs_[idx],
                                                   &tag->resp_tensor_bufs_[idx]);
      }
    } else {
      uint16_t err_len =
          std::min(UINT16_MAX, (int) (s.error_message().length()));
      memcpy(tag->resp_header_buf_.data_ + 22, &err_len, 2);
      memcpy(tag->resp_header_buf_.data_ + 32, s.error_message().c_str(),
             err_len);
      tag->StartRespWithTensor();
      return;
    }

    tag->StartRespWithFuseTensor();
    //LOG(INFO) << "step " << static_cast<FuseRecvTensorRequest*>(request)->step_id() << " has been sent.";
  };
  // used for zero copy sending tensor, unref Tensor object after star send done
  tag->clear_ = std::move(clear);
  tag->send_resp_ = std::move(done);
}

const char * SeastarServerTag::HEADER_SIGN = "CAFEBABE";

SeastarServerTag::SeastarServerTag(seastar::channel* seastar_channel,
                                   SeastarWorkerService* seastar_worker_service)
    : seastar_channel_(seastar_channel),
      seastar_worker_service_(seastar_worker_service),
      tag_write_start_micros(0) {}

SeastarServerTag::~SeastarServerTag() {}

// Called by seastar engine, call the handler.
void SeastarServerTag::RecvReqDone(const Status& s) {
  if (!s.ok()) {
    this->send_resp_(s);
    // TODO(handle clear)
    return;
  }

  SeastarWorkerService::HandleRequestFunction handle =
      seastar_worker_service_->GetHandler(method_);
  (seastar_worker_service_->*handle)(this);
}

// Called by seastar engine.
void SeastarServerTag::SendRespDone() {
  clear_(Status());
  delete this;
}

// Serialize and send response.
void SeastarServerTag::ProcessDone(Status s) {
  // LOG(INFO) << "enter seastarServerTag::ProcessDone";
  send_resp_(s);
}

uint64_t SeastarServerTag::GetRequestBodySize() { return req_body_buf_.len_; }

char* SeastarServerTag::GetRequestBodyBuffer() { return req_body_buf_.data_; }

void SeastarServerTag::StartResp() { seastar_channel_->put(ToUserPacket()); }

void SeastarServerTag::StartRespWithTensor() {
  seastar_channel_->put(ToUserPacketWithTensor());
}

void SeastarServerTag::StartRespWithFuseTensor() {
  seastar_channel_->put(ToUserPacketWithFuseTensors());
}

void SeastarServerTag::InitResponseTensorBufs(int32_t resp_tensor_count) {
  resp_tensor_count_ = resp_tensor_count;
  resp_message_bufs_.resize(resp_tensor_count);
  resp_tensor_bufs_.resize(resp_tensor_count);
}

bool SeastarServerTag::IsRecvTensor() {
  return method_ == SeastarWorkerServiceMethod::kRecvTensor;
}

bool SeastarServerTag::IsFuseRecvTensor() {
  return method_ == SeastarWorkerServiceMethod::kFuseRecvTensor;
}

seastar::user_packet* SeastarServerTag::ToUserPacket() {
  seastar::net::fragment respHeader{resp_header_buf_.data_,
                                    resp_header_buf_.len_};
  seastar::net::fragment respBody{resp_body_buf_.data_, resp_body_buf_.len_};

  std::vector<seastar::net::fragment> frags = {respHeader, respBody};
  auto* up = new seastar::user_packet;
  up->_fragments = frags;
  up->_done = [this]() { this->SendRespDone(); };
  return up;
}

seastar::user_packet* SeastarServerTag::ToUserPacketWithTensor() {
  auto up = new seastar::user_packet;
  std::vector<seastar::net::fragment> frags;
  frags.emplace_back(
      seastar::net::fragment{resp_header_buf_.data_, resp_header_buf_.len_});

  frags.emplace_back(
      seastar::net::fragment{resp_message_buf_.data_, resp_message_buf_.len_});

  if (resp_tensor_buf_.len_ > 0) {
    frags.emplace_back(
        seastar::net::fragment{resp_tensor_buf_.data_, resp_tensor_buf_.len_});
  }
  up->_fragments = frags;
  up->_done = [this]() { this->SendRespDone(); };
  return up;
}

std::vector<seastar::user_packet*> SeastarServerTag::ToUserPacketWithFuseTensors() {
  int64 total_len = 0;
  std::vector<seastar::user_packet*> ret;
  auto up = new seastar::user_packet;

  std::vector<seastar::net::fragment> frags;
  frags.emplace_back(
      seastar::net::fragment {resp_header_buf_.data_, resp_header_buf_.len_});
  total_len += resp_header_buf_.len_;

  // For fuse recv / zero copy run graph, if error happens 'resp_tensor_count_'
  // is zero as when it is inited, so no tensor can be sent.
  for (auto i = 0; i < resp_tensor_count_; ++i) {
    if (frags.size() > IOV_MAX / 2) {//max fragments
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
}  // namespace tensorflow

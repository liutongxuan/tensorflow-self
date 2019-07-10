#include "tensorflow/contrib/seastar/seastar_client_tag.h"

#include "tensorflow/contrib/seastar/seastar_message.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace {

void ProcessCallOptions(SeastarClientTag* tag) {
  if (tag->call_opts_ != nullptr) {
    if (tag->call_opts_->GetTimeout() > 0) {
      tag->timeout_in_ms_ = tag->call_opts_->GetTimeout();
    }
  }
}

}  // namespace

void InitSeastarClientTag(protobuf::Message* request,
                          protobuf::Message* response, StatusCallback done,
                          SeastarClientTag* tag, CallOptions* call_opts) {
  tag->req_body_buf_.len_ = request->ByteSize();
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_]();
  request->SerializeToArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  tag->req_header_buf_.len_ = SeastarClientTag::HEADER_SIZE;
  tag->req_header_buf_.data_ = new char[SeastarClientTag::HEADER_SIZE];

  memcpy(tag->req_header_buf_.data_, SeastarClientTag::HEADER_SIGN, 8);
  memcpy(tag->req_header_buf_.data_ + 8, &tag, 8);
  memcpy(tag->req_header_buf_.data_ + 16, &tag->method_, 4);
  memcpy(tag->req_header_buf_.data_ + 24, &tag->req_body_buf_.len_, 8);

  StatusCallback wrapper_done = std::bind(
      [response, tag](const StatusCallback& done, const Status& s) {
        if (!s.ok()) {
          if (tag->method_ == SeastarWorkerServiceMethod::kLogging ||
              tag->method_ == SeastarWorkerServiceMethod::kTracing) {
            // Logging & Tracing in worker.cc is UNIMPLEMENTED, ignore the error
          } else {
            // Debugging info
            LOG(INFO) << "RPC's status is not ok. status code=" << s.code()
                      << ", err msg=" << s.error_message().c_str();
          }
        } else {
          response->ParseFromArray(tag->resp_body_buf_.data_,
                                   tag->resp_body_buf_.len_);
        }
        done(s);
        delete tag;
      },
      std::move(done), std::placeholders::_1);

  tag->done_ = std::move(wrapper_done);
  tag->call_opts_ = call_opts;
  ProcessCallOptions(tag);
}

void InitSeastarClientTag(protobuf::Message* request,
                          SeastarTensorResponse* response, StatusCallback done,
                          SeastarClientTag* tag, CallOptions* call_opts) {
  tag->req_body_buf_.len_ = request->ByteSize();
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_]();
  request->SerializeToArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  tag->req_header_buf_.len_ = SeastarClientTag::HEADER_SIZE;
  tag->req_header_buf_.data_ = new char[SeastarClientTag::HEADER_SIZE];

  memcpy(tag->req_header_buf_.data_, SeastarClientTag::HEADER_SIGN, 8);
  memcpy(tag->req_header_buf_.data_ + 8, &tag, 8);
  memcpy(tag->req_header_buf_.data_ + 16, &tag->method_, 4);
  // Ignore the status segment in request
  // memcpy(tag->req_header_buf_.data_ + 20, &tag->status_, 2);
  memcpy(tag->req_header_buf_.data_ + 24, &tag->req_body_buf_.len_, 8);

  ParseMessageCallback wrapper_parse_message = [request, response, tag]() {
    SeastarMessage sm;
    SeastarMessage::DeserializeMessage(&sm, tag->resp_message_buf_.data_);

    response->SetIsDead(sm.is_dead_);
    response->set_send_start_micros(sm.send_start_micros_);
    response->SetDataType(sm.data_type_);
    bool can_memcpy = DataTypeCanUseMemcpy(sm.data_type_);

    if (can_memcpy) {
      if (response->GetDevice()->tensorflow_gpu_device_info() &&
          (!response->GetOnHost())) {
        AllocatorAttributes alloc_attrs;
        alloc_attrs.set_gpu_compatible(true);
        alloc_attrs.set_on_host(true);
        Allocator* alloc = response->GetDevice()->GetAllocator(alloc_attrs);
        Tensor cpu_copy(alloc, sm.data_type_, sm.tensor_shape_);

        tag->resp_tensor_buf_.data_ =
            reinterpret_cast<char*>(DMAHelper::base(&cpu_copy));
        tag->resp_tensor_buf_.len_ = sm.tensor_bytes_;
        tag->resp_tensor_buf_.owned_ = false;

        response->SetTensor(cpu_copy);

      } else {
        // let tag.resp_tensor_buf_.data directly points to the tensor addreess
        // allocated by tensroflow allocator, later in seastar cleint, tensor
        // content will be filled in by memcpy.
        Tensor val(response->GetAlloc(), sm.data_type_, sm.tensor_shape_);
        tag->resp_tensor_buf_.data_ =
            reinterpret_cast<char*>(DMAHelper::base(&val));
        tag->resp_tensor_buf_.len_ = sm.tensor_bytes_;
        tag->resp_tensor_buf_.owned_ = false;

        response->SetTensor(val);
      }
    } else {
      tag->resp_tensor_buf_.len_ = sm.tensor_bytes_;
      tag->resp_tensor_buf_.data_ = new char[tag->resp_tensor_buf_.len_]();
    }

    return Status();
  };
  tag->parse_message_ = std::move(wrapper_parse_message);

  StatusCallback wrapper_done = std::bind(
      [response, tag](const StatusCallback& done, const Status& s) {
        if (!s.ok()) {
          VLOG(1) << "wrapper_done, status not ok. status code=" << s.code()
                  << ", err msg=" << s.error_message().c_str();
          done(s);
          delete tag;
          return;
        }

        bool can_memcpy = DataTypeCanUseMemcpy(response->GetDataType());
        if (can_memcpy) {
          if (response->GetDevice()->tensorflow_gpu_device_info() &&
              (!response->GetOnHost())) {
            auto* gpu_copy =
                new Tensor(response->GetAlloc(), response->GetTensor().dtype(),
                           response->GetTensor().shape());
            DeviceContext* recv_dev_context = response->GetDevice()
                ->tensorflow_gpu_device_info()
                ->default_context;
            recv_dev_context->CopyCPUTensorToDevice(
                &response->GetTensor(), response->GetDevice(), gpu_copy,
                [gpu_copy, response, done, tag](const Status& s) {
                  CHECK(s.ok()) << "copy tensor to gpu sync";
                  response->SetTensor(*gpu_copy);
                  done(s);
                  delete gpu_copy;
                  delete tag;
                });
          } else {
            done(s);
            delete tag;
          }
        } else {
          // could not memcopy
          ParseProtoUnlimited(&response->GetTensorProto(),
                              tag->resp_tensor_buf_.data_,
                              tag->resp_tensor_buf_.len_);
          Tensor val;
          Status status = response->GetDevice()->MakeTensorFromProto(
              response->GetTensorProto(), response->GetAllocAttributes(), &val);
          CHECK(status.ok()) << "make cpu tensor from proto.";
          response->SetTensor(val);
          done(status);
          delete tag;
        }
      },
      std::move(done), std::placeholders::_1);

  tag->done_ = std::move(wrapper_done);
  tag->call_opts_ = call_opts;
  ProcessCallOptions(tag);
}

void InitSeastarClientTag(protobuf::Message *request,
                          SeastarFuseTensorResponse *response,
                          StatusCallback done,
                          SeastarClientTag *tag,
                          CallOptions *call_opts) {
  tag->req_body_buf_.len_ = request->ByteSize();
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_]();
  request->SerializeToArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  tag->req_header_buf_.len_ = SeastarClientTag::HEADER_SIZE;
  tag->req_header_buf_.data_ = new char[SeastarClientTag::HEADER_SIZE];

  memcpy(tag->req_header_buf_.data_, SeastarClientTag::HEADER_SIGN, 8);
  memcpy(tag->req_header_buf_.data_ + 8, &tag, 8);
  memcpy(tag->req_header_buf_.data_ + 16, &tag->method_, 4);
  memcpy(tag->req_header_buf_.data_ + 24, &tag->req_body_buf_.len_, 8);

  ParseFuseMessageCallback wrapper_parse_message
      = [request, response, tag](int idx, const char *tensor_msg, size_t len) {
        CHECK_EQ(SeastarMessage::kMessageTotalBytes, len);
        SeastarMessage sm;
        SeastarMessage::DeserializeMessage(&sm, tensor_msg);

        response->SetIsDeadByIndex(idx, sm.is_dead_);
        response->SetDataTypeByIndex(idx, sm.data_type_);
        if (idx == 0) {
          response->set_send_start_micros(sm.send_start_micros_);
        }

        if (sm.tensor_bytes_ == 0){return Status();}
        bool can_memcpy = DataTypeCanUseMemcpy(sm.data_type_);

        if (can_memcpy) {
          if (response->GetDevice()->tensorflow_gpu_device_info() &&
              (!response->GetOnHost())) {
            return errors::Internal("No GPU device in process");
          } else {
            Tensor val(response->GetAlloc(), sm.data_type_, sm.tensor_shape_);
            tag->resp_tensor_bufs_[idx].data_ =
                reinterpret_cast<char *>(DMAHelper::base(&val));
            tag->resp_tensor_bufs_[idx].len_ = sm.tensor_bytes_;
            tag->resp_tensor_bufs_[idx].owned_ = false;
            response->SetTensorByIndex(idx, val);
          }
        } else {
          //set response later in done
          tag->resp_tensor_bufs_[idx].len_ = sm.tensor_bytes_;
          tag->resp_tensor_bufs_[idx].data_ =
              new char[tag->resp_tensor_bufs_[idx].len_]();
        }

        return Status();
      };
  tag->parse_fuse_message_ = std::move(wrapper_parse_message);

  StatusCallback wrapper_done
      = std::bind([response, tag](StatusCallback done, const Status &s) {
                    if (!s.ok()) {
                      LOG(ERROR) << "wrapper_done, status not ok. status code=" << s.code()
                                 << ", err msg=" << s.error_message().c_str();
                      done(s);
                      delete tag;
                      return;
                    }

                    int tensor_count = tag->req_tensor_count_;
                    int *resp_tensor_counter = new int(tensor_count);

                    for (int idx = 0; idx < tensor_count; ++idx) {
                      bool can_memcpy =
                          DataTypeCanUseMemcpy(response->GetDataTypeByIndex(idx));
                      if (can_memcpy) {
                        if (response->GetDevice()->tensorflow_gpu_device_info() &&
                            (!response->GetOnHost())) {
                          done(errors::Internal("No GPU device in process"));
                          // delete tag;
                          // It may be not safe to delete tag here, just abort here.
                          abort();
                        } else {
                          if (__sync_sub_and_fetch(resp_tensor_counter, 1) == 0) {
                            delete resp_tensor_counter;
                            done(s);
                            delete tag;
                          }
                        }
                      } else {
                        // Could not memory copy.
                        ParseProtoUnlimited(&response->GetTensorProtoByIndex(idx),
                                            tag->resp_tensor_bufs_[idx].data_,
                                            tag->resp_tensor_bufs_[idx].len_);
                        Tensor val;
                        Status status = response->GetDevice()->MakeTensorFromProto(
                            response->GetTensorProtoByIndex(idx),
                            response->GetAllocAttributes(), &val);
                        CHECK(status.ok()) << "Make cpu tensor from proto.";
                        response->SetTensorByIndex(idx, val);
                        if (__sync_sub_and_fetch(resp_tensor_counter, 1) == 0) {
                          delete resp_tensor_counter;
                          done(status);
                          delete tag;
                        }
                      }
                    } // End for loop of the fuse count.
                  },
                  std::move(done),
                  std::placeholders::_1);

  tag->done_ = std::move(wrapper_done);
  tag->call_opts_ = call_opts;
  ProcessCallOptions(tag);
}

const char * SeastarClientTag::HEADER_SIGN = "DEADBEEF";

SeastarClientTag::SeastarClientTag(
    tensorflow::SeastarWorkerServiceMethod method, WorkerEnv *env)
    : method_(method),
      env_(env),
      status_(0),
      resp_err_msg_len_(0),
      call_opts_(nullptr),
      timeout_in_ms_(0),
      tag_read_start_micros(0),
      req_tensor_count_(1){
  resp_message_buf_.len_ = SeastarMessage::kMessageTotalBytes;
  resp_message_buf_.data_ = new char[resp_message_buf_.len_];
}

SeastarClientTag::SeastarClientTag(tensorflow::SeastarWorkerServiceMethod method,
                                   WorkerEnv *env, int req_tensor_count)
    : method_(method),
      env_(env),
      status_(0),
      resp_err_msg_len_(0),
      req_tensor_count_(req_tensor_count),
      call_opts_(nullptr),
      timeout_in_ms_(0),
      tag_read_start_micros(0){
}

SeastarClientTag::~SeastarClientTag() {}

void SeastarClientTag::StartReq(seastar::channel* seastar_channel) {
  seastar_channel->put(ToUserPacket());
}

bool SeastarClientTag::IsRecvTensor() {
  return method_ == SeastarWorkerServiceMethod::kRecvTensor;
}

Status SeastarClientTag::ParseMessage() { return parse_message_(); }

void SeastarClientTag::Schedule(std::function<void()> f) {
  env_->compute_pool->Schedule(std::move(f));
}

void SeastarClientTag::RecvRespDone(Status s) {
  Schedule([this, s]() { done_(s); });
}

uint64_t SeastarClientTag::GetResponseBodySize() { return resp_body_buf_.len_; }

char* SeastarClientTag::GetResponseBodyBuffer() { return resp_body_buf_.data_; }

uint64_t SeastarClientTag::GetResponseMessageSize() {
  return resp_message_buf_.len_;
}

char* SeastarClientTag::GetResponseMessageBuffer() {
  return resp_message_buf_.data_;
}

uint64_t SeastarClientTag::GetResponseTensorSize() {
  return resp_tensor_buf_.len_;
}

char* SeastarClientTag::GetResponseTensorBuffer() {
  return resp_tensor_buf_.data_;
}

bool SeastarClientTag::IsFuseRecvTensor() {
  return method_ == SeastarWorkerServiceMethod::kFuseRecvTensor;
}

Status SeastarClientTag::ParseFuseMessage(int idx,
                                            const char *tensor_msg,
                                            size_t len) {
  return parse_fuse_message_(idx, tensor_msg, len);
}

void SeastarClientTag::HandleResponse(Status s) {
  done_(s);
}

void SeastarClientTag::ScheduleProcess(std::function<void()> f) {
  env_->compute_pool->Schedule(std::move(f));
}

// tensor size
uint64_t SeastarClientTag::GetResponseTensorSize(int idx) {
  return resp_tensor_bufs_[idx].len_;
}

// tensor buffer
char* SeastarClientTag::GetResponseTensorBuffer(int idx) {
  return resp_tensor_bufs_[idx].data_;
}

void SeastarClientTag::InitTensorBuffers(int fuse_count){
  resp_tensor_bufs_.resize(fuse_count);
}

seastar::user_packet* SeastarClientTag::ToUserPacket() {
  seastar::net::fragment reqHeader{req_header_buf_.data_, req_header_buf_.len_};
  seastar::net::fragment reqBody{req_body_buf_.data_, req_body_buf_.len_};

  std::vector<seastar::net::fragment> frags = {reqHeader, reqBody};
  auto* up = new seastar::user_packet;
  up->_fragments = frags;
  up->_done = [] {};
  return up;
}

}  // namespace tensorflow

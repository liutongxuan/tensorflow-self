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
                          protobuf::Message* response,
                          StatusCallback done,
                          SeastarClientTag* tag,
                          CallOptions* call_opts) {
  tag->req_body_buf_.len_ = request->ByteSize();
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_]();
  request->SerializeToArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  tag->req_header_buf_.len_ = SeastarClientTag::kHeaderSize;
  tag->req_header_buf_.data_ = new char[SeastarClientTag::kHeaderSize]();

  memcpy(tag->req_header_buf_.data_, "AAAA", 4);
  memcpy(tag->req_header_buf_.data_ + SeastarClientTag::kMethodIndex,
         &tag->method_, 4);
  memcpy(tag->req_header_buf_.data_ + SeastarClientTag::kTagIndex, &tag, 8);
  // Ignore the status and user_data segment.
  memcpy(tag->req_header_buf_.data_ + SeastarClientTag::kPayloadLenIndex,
         &tag->req_body_buf_.len_, 8);

  StatusCallback wrapper_done
    = std::bind([response, tag](StatusCallback done,
                                const Status& s) {
                  // internal error, that we dont need to parse the response,
                  // reponse is nullptr
                  if (s.code() != error::INTERNAL) {
                    response->ParseFromArray(tag->resp_body_buf_.data_,
                                            tag->resp_body_buf_.len_);
                  }
                  if (!s.ok()) {
                    if (tag->method_ == SeastarWorkerServiceMethod::kLogging ||
                        tag->method_ == SeastarWorkerServiceMethod::kTracing) {
                      // Logging & Tracing in worker.cc is UNIMPLEMENTED, ignore the error
                    } else {
                      // Debugging info
                      LOG(INFO) << "RPC's status is not ok. status code=" << s.code()
                                << ", err msg=" << s.error_message().c_str();
                    }
                  }
                  done(s);
                  delete tag;
                },
                std::move(done),
                std::placeholders::_1);
  tag->done_ = std::move(wrapper_done);
  tag->call_opts_ = call_opts;
  tag->ProcessCallOptions();
}

void InitSeastarClientTag(protobuf::Message* request,
                          SeastarTensorResponse* response,
                          StatusCallback done,
                          SeastarClientTag* tag,
                          CallOptions* call_opts) {
  tag->req_body_buf_.len_ = request->ByteSize();
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_]();
  request->SerializeToArray(tag->req_body_buf_.data_,
                            tag->req_body_buf_.len_);


  tag->req_header_buf_.len_ = SeastarClientTag::kHeaderSize;
  tag->req_header_buf_.data_ = new char[SeastarClientTag::kHeaderSize]();

  memcpy(tag->req_header_buf_.data_, "AAAA", 4);
  memcpy(tag->req_header_buf_.data_ + SeastarClientTag::kMethodIndex,
         &tag->method_, 4);
  memcpy(tag->req_header_buf_.data_ + SeastarClientTag::kTagIndex, &tag, 8);
  // Ignore the status and user_data segment.
  memcpy(tag->req_header_buf_.data_ + SeastarClientTag::kPayloadLenIndex,
         &tag->req_body_buf_.len_, 8);

  ParseMessageCallback wrapper_parse_message
    = [request, response, tag] (int idx, const char* tensor_msg, size_t len) {
      CHECK_EQ(SeastarMessage::kMessageTotalBytes, len);
      SeastarMessage sm;
      SeastarMessage::DeserializeMessage(&sm, tensor_msg);

      response->SetIsDead(sm.is_dead_);
      response->SetDataType(sm.data_type_);
      bool can_memcpy = DataTypeCanUseMemcpy(sm.data_type_);

      if (can_memcpy) {
        if (response->GetDevice()->tensorflow_gpu_device_info() &&
            (!response->GetOnHost())) {
 #if GOOGLE_CUDA
          // LOG(INFO) << "parse msg, can memcpy and on GPU";
          // dst tensor on gpu
          Allocator* alloc = ProcessState::singleton()->GetCUDAHostAllocator(0);
          Tensor cpu_copy(alloc, sm.data_type_, sm.tensor_shape_);

          tag->resp_tensor_bufs_[idx].data_ = reinterpret_cast<char*>(DMAHelper::base(&cpu_copy));
          tag->resp_tensor_bufs_[idx].len_ =  sm.tensor_bytes_;
          tag->resp_tensor_bufs_[idx].owned_ = false;

          response->SetTensorByIndex(idx, cpu_copy);
#else
          return errors::Internal("No GPU device in process");
#endif

        } else {
          Tensor val(response->GetAlloc(), sm.data_type_, sm.tensor_shape_);
          tag->resp_tensor_bufs_[idx].data_ = reinterpret_cast<char*>(DMAHelper::base(&val));
          tag->resp_tensor_bufs_[idx].len_ =  sm.tensor_bytes_;
          tag->resp_tensor_bufs_[idx].owned_ = false;

          response->SetTensorByIndex(idx, val);
        }
      } else {
        tag->resp_tensor_bufs_[idx].len_ = sm.tensor_bytes_;
        tag->resp_tensor_bufs_[idx].data_ = new char[tag->resp_tensor_bufs_[idx].len_]();
      }

      return Status();
  };
  tag->parse_message_ = std::move(wrapper_parse_message);

  StatusCallback wrapper_done
    = std::bind([response, tag](StatusCallback done,
                                const Status& s) {
                  if (!s.ok()) {
                    LOG(ERROR) << "wrapper_done, status not ok. status code=" << s.code()
                               << ", err msg=" << s.error_message().c_str();
                    done(s);
                    delete tag;
                    return;
                  }

                  int resp_tensor_count = tag->resp_tensor_count_;
                  int *resp_tensor_counter = new int(resp_tensor_count);

                  for (int idx = 0; idx < resp_tensor_count; ++idx) {
                    bool can_memcpy = DataTypeCanUseMemcpy(response->GetDataTypeByIndex(idx));
                    if (can_memcpy) {
                      if (response->GetDevice()->tensorflow_gpu_device_info() &&
                          (!response->GetOnHost())) {
#if GOOGLE_CUDA
                        Tensor* gpu_copy = new Tensor(response->GetAlloc(),
                                                      response->GetTensorByIndex(idx).dtype(),
                                                      response->GetTensorByIndex(idx).shape());
                        GPUUtil::CopyCPUTensorToGPU(&response->GetTensorByIndex(idx),
                                                    response->GetDevice()->tensorflow_gpu_device_info()->default_context,
                                                    response->GetDevice(),
                                                    gpu_copy,
                                                    [gpu_copy, response, done, tag, resp_tensor_counter, idx](const Status& s) {
                                                      CHECK(s.ok()) << "copy tensor to gpu sync";
                                                      response->SetTensorByIndex(idx, *gpu_copy);
                                                      delete gpu_copy;
                                                      if (__sync_sub_and_fetch(resp_tensor_counter, 1) == 0) {
                                                        delete resp_tensor_counter;
                                                        done(s);
                                                        delete tag;
                                                      }
                                                    });
#else
                        done(errors::Internal("No GPU device in process"));
                        // delete tag;
                        // It may be not safe to delete tag here, just abort here.
                        abort();
#endif
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
                    }
                  } // End for loop of the fuse count.
                },
                std::move(done),
                std::placeholders::_1);

  tag->done_ = std::move(wrapper_done);
  tag->call_opts_ = call_opts;
  tag->ProcessCallOptions();
}

void InitSeastarClientTag(protobuf::Message* request,
                       SeastarFuseTensorResponse* response,
                       StatusCallback done,
                       SeastarClientTag* tag,
                       CallOptions* call_opts) {
  tag->req_body_buf_.len_ = request->ByteSize();
  tag->req_body_buf_.data_ = new char[tag->req_body_buf_.len_]();
  request->SerializeToArray(tag->req_body_buf_.data_, tag->req_body_buf_.len_);

  tag->req_header_buf_.len_ = SeastarClientTag::kHeaderSize;
  tag->req_header_buf_.data_ = new char[SeastarClientTag::kHeaderSize]();

  memcpy(tag->req_header_buf_.data_, "AAAA", 4);
  memcpy(tag->req_header_buf_.data_ + SeastarClientTag::kMethodIndex,
         &tag->method_, 4);
  memcpy(tag->req_header_buf_.data_ + SeastarClientTag::kTagIndex, &tag, 8);
  // Ignore the status and user_data segment.
  memcpy(tag->req_header_buf_.data_ + SeastarClientTag::kPayloadLenIndex,
         &tag->req_body_buf_.len_, 8);

  ParseMessageCallback wrapper_parse_message
    = [request, response, tag] (int idx, const char* tensor_msg, size_t len) {
      CHECK_EQ(SeastarMessage::kMessageTotalBytes, len);
      SeastarMessage sm;
      SeastarMessage::DeserializeMessage(&sm, tensor_msg);

      response->SetIsDeadByIndex(idx, sm.is_dead_);
      response->SetDataTypeByIndex(idx, sm.data_type_);
      bool can_memcpy = DataTypeCanUseMemcpy(sm.data_type_);

      if (can_memcpy) {
        if (response->GetDevice()->tensorflow_gpu_device_info() &&
            (!response->GetOnHost())) {
 #if GOOGLE_CUDA
          Allocator* alloc = ProcessState::singleton()->GetCUDAHostAllocator(0);
          Tensor cpu_copy(alloc, sm.data_type_, sm.tensor_shape_);

          tag->resp_tensor_bufs_[idx].data_ = reinterpret_cast<char*>(DMAHelper::base(&cpu_copy));
          tag->resp_tensor_bufs_[idx].len_ =  sm.tensor_bytes_;
          tag->resp_tensor_bufs_[idx].owned_ = false;

          response->SetTensorByIndex(idx, cpu_copy);
#else
          return errors::Internal("No GPU device in process");
#endif
        } else {
          Tensor val(response->GetAlloc(), sm.data_type_, sm.tensor_shape_);
          tag->resp_tensor_bufs_[idx].data_ = reinterpret_cast<char*>(DMAHelper::base(&val));
          tag->resp_tensor_bufs_[idx].len_ =  sm.tensor_bytes_;
          tag->resp_tensor_bufs_[idx].owned_ = false;

          response->SetTensorByIndex(idx, val);
        }
      } else {
        tag->resp_tensor_bufs_[idx].len_ = sm.tensor_bytes_;
        tag->resp_tensor_bufs_[idx].data_ = new char[tag->resp_tensor_bufs_[idx].len_]();
      }

      return Status();
  };
  tag->parse_message_ = std::move(wrapper_parse_message);

  StatusCallback wrapper_done
    = std::bind([response, tag](StatusCallback done,
                                const Status& s) {
                  if (!s.ok()) {
                    LOG(ERROR) << "wrapper_done, status not ok. status code=" << s.code()
                               << ", err msg=" << s.error_message().c_str();
                    done(s);
                    delete tag;
                    return;
                  }

                  int resp_tensor_count = tag->resp_tensor_count_;
                  int *resp_tensor_counter = new int(resp_tensor_count);

                  for (int idx = 0; idx < resp_tensor_count; ++idx) {
                    bool can_memcpy = DataTypeCanUseMemcpy(response->GetDataTypeByIndex(idx));
                    if (can_memcpy) {
                      if (response->GetDevice()->tensorflow_gpu_device_info() &&
                          (!response->GetOnHost())) {
#if GOOGLE_CUDA
                        Tensor* gpu_copy = new Tensor(response->GetAlloc(),
                                                      response->GetTensorByIndex(idx).dtype(),
                                                      response->GetTensorByIndex(idx).shape());
                        GPUUtil::CopyCPUTensorToGPU(&response->GetTensorByIndex(idx),
                                                    response->GetDevice()->tensorflow_gpu_device_info()->default_context,
                                                    response->GetDevice(),
                                                    gpu_copy,
                                                    [gpu_copy, response, done, tag, resp_tensor_counter, idx](const Status& s) {
                                                      CHECK(s.ok()) << "copy tensor to gpu sync";
                                                      response->SetTensorByIndex(idx, *gpu_copy);
                                                      delete gpu_copy;
                                                      if (__sync_sub_and_fetch(resp_tensor_counter, 1) == 0) {
                                                        delete resp_tensor_counter;
                                                        done(s);
                                                        delete tag;
                                                      }
                                                    });
#else
                        done(errors::Internal("No GPU device in process"));
                        // delete tag;
                        // It may be not safe to delete tag here, just abort here.
                        abort();
#endif
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
  tag->ProcessCallOptions();
}

SeastarClientTag::SeastarClientTag(tensorflow::SeastarWorkerServiceMethod method,
                             WorkerEnv* env, int resp_tensor_count,
                             int req_tensor_count)
  : method_(method), status_(0), err_msg_len_(0),
    req_tensor_count_(req_tensor_count),
    resp_tensor_count_(resp_tensor_count),
    req_message_bufs_(req_tensor_count),
    req_tensor_bufs_(req_tensor_count),
    resp_tensor_bufs_(resp_tensor_count),
    parse_meta_data_(nullptr), parse_message_(nullptr), done_(nullptr),
    env_(env), call_opts_(nullptr),
    fail_fast_(false), timeout_in_ms_(0),
    resp_packet_pos_(nullptr), resp_packet_len_(0) {}

SeastarClientTag::~SeastarClientTag() {
  delete [] req_header_buf_.data_;
  delete [] req_body_buf_.data_;
  delete [] resp_body_buf_.data_;

  for (int i = 0; i < resp_tensor_count_; ++i) {
    if (resp_tensor_bufs_[i].owned_) {
      delete [] resp_tensor_bufs_[i].data_;
    }
  }

  for (uint64_t i = 0; i < req_tensor_count_; ++i) {
    delete [] req_message_bufs_[i].data_;
    if (req_tensor_bufs_[i].owned_) {
      delete [] req_tensor_bufs_[i].data_;
    }
  }
}

bool SeastarClientTag::IsRecvTensor() {
  return method_ == SeastarWorkerServiceMethod::kRecvTensor
    || method_ == SeastarWorkerServiceMethod::kFuseRecvTensor;
}

Status SeastarClientTag::ParseTensorMessage(int idx, const char* tensor_msg, size_t len) {
  return parse_message_(idx, tensor_msg, len);
}

void SeastarClientTag::ScheduleProcess(std::function<void()> f) {
  env_->compute_pool->Schedule(std::move(f));
}

void SeastarClientTag::RepeatedlyParseTensors(char* p) {
  for (auto i = 0; i < resp_tensor_count_; ++i) {
    ParseTensorMessage(i, p, SeastarMessage::kMessageTotalBytes);
    auto tensor_size = GetResponseTensorSize(i);
    auto tensor_buffer = GetResponseTensorBuffer(i);
    memcpy(tensor_buffer, p + SeastarMessage::kMessageTotalBytes, tensor_size);

    p += SeastarMessage::kMessageTotalBytes + tensor_size;
  }
}

Status SeastarClientTag::ParseResponse() {
  memcpy(&status_, resp_packet_pos_ + SeastarServerTag::kStatusIndex, 4);
  if (status_ != 0) {
    std::string error_msg
      = std::string(resp_packet_pos_ + SeastarServerTag::kHeaderSize,
                    resp_packet_len_ - SeastarServerTag::kHeaderSize);
    if (error_msg.empty()) {
      error_msg = "Empty error msg.";
    }

    return tensorflow::Status(static_cast<tensorflow::error::Code>(status_),
                              error_msg);
  }

  if (IsRecvTensor()) {
    RepeatedlyParseTensors(resp_packet_pos_ + SeastarServerTag::kHeaderSize);

  } else {
    memcpy(&resp_body_buf_.len_,
           resp_packet_pos_ + SeastarClientTag::kPayloadLenIndex, 8);
    resp_body_buf_.data_ = new char[resp_body_buf_.len_];

    memcpy(resp_body_buf_.data_,
           resp_packet_pos_ + SeastarServerTag::kHeaderSize,
           resp_body_buf_.len_);
  }

  return tensorflow::Status();
}

void SeastarClientTag::HandleResponse(Status s) {
  done_(s);
}

// payload size for non-tensor response
uint64_t SeastarClientTag::GetResponseBodySize() {
  return resp_body_buf_.len_;
}

// payload buffer for non-tensor response
char* SeastarClientTag::GetResponseBodyBuffer() {
  return resp_body_buf_.data_;
}

// tensor size
uint64_t SeastarClientTag::GetResponseTensorSize(int idx) {
  return resp_tensor_bufs_[idx].len_;
}

// tensor buffer
char* SeastarClientTag::GetResponseTensorBuffer(int idx) {
  return resp_tensor_bufs_[idx].data_;
}

}  // namespace tensorflow

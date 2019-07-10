#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_TAG_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_TAG_H_

#include "tensorflow/contrib/seastar/seastar_tensor_coding.h"
#include "tensorflow/contrib/seastar/seastar_worker_service.h"
#include "tensorflow/contrib/seastar/seastar_worker_service_method.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "third_party/seastar/core/channel.hh"
#include "third_party/seastar/core/packet_queue.hh"
#include "third_party/seastar/core/temporary_buffer.hh"

namespace tensorflow {

typedef std::function<Status()> ParseMessageCallback;

class SeastarClientTag {
 public:
  // Client Header 32B:
  // |ID:8B|tag:8B|method:4B|reserve:4B|body_len:8B|
  static const size_t kMethodIndex = 4;
  static const size_t kTagIndex = 8;
  static const size_t kStatusIndex = 16;
  static const size_t kUserDataIndex = 20;
  static const size_t kPayloadLenIndex = 24;
  static const size_t kHeaderSize = 32;

  SeastarClientTag(tensorflow::SeastarWorkerServiceMethod method,
                   WorkerEnv* env, int resp_tensor_count,
                   int req_tensor_count);
  virtual ~SeastarClientTag();

  SeastarClientTag(tensorflow::SeastarWorkerServiceMethod method,
                   WorkerEnv* env, int resp_tesnsor_count,
                   int req_tensor_count);
  virtual ~SeastarClientTag();

  bool IsRecvTensor();
  Status ParseTensorMessage(int idx, const char* tensor_msg, size_t len);

  Status ParseResponse();
  void RepeatedlyParseTensors(char* p);
  void HandleResponse(Status s);
  void ScheduleProcess(std::function<void()> f);

  uint64_t GetResponseBodySize();
  char* GetResponseBodyBuffer();

  uint64_t GetResponseTensorSize(int idx);
  char* GetResponseTensorBuffer(int idx);

  void ProcessCallOptions();

 protected:
  SeastarWorkerServiceMethod method_;
  int32 status_;
  uint64_t err_msg_len_;
  int req_tensor_count_;
  int resp_tensor_count_;

  SeastarBuf req_header_buf_;
  SeastarBuf req_body_buf_;
  SeastarBuf resp_body_buf_;

  std::vector<SeastarBuf> req_message_bufs_;
  std::vector<SeastarBuf> req_tensor_bufs_;
  std::vector<SeastarBuf> resp_tensor_bufs_;

  ParseMetaDataCallback parse_meta_data_;
  ParseMessageCallback parse_message_;
  StatusCallback done_;

  WorkerEnv* env_;
  CallOptions* call_opts_;

  bool fail_fast_;
  int timeout_in_ms_;
  char* resp_packet_pos_; // Not owned.
  int64_t resp_packet_len_;
};

void InitSeastarClientTag(protobuf::Message* request,
                          protobuf::Message* response, StatusCallback done,
                          SeastarClientTag* tag, CallOptions* call_opts);

void InitSeastarClientTag(protobuf::Message* request,
                          SeastarTensorResponse* response, StatusCallback done,
                          SeastarClientTag* tag, CallOptions* call_opts);

void InitSeastarClientTag(protobuf::Message* request,
                          SeastarFuseTensorResponse* response,
                          SeastatusCallback done,
                          SeastarClientTag* tag,
                          CallOptions* call_opts);

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_CLIENT_TAG_H_

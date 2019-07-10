#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_SERVER_TAG_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_SERVER_TAG_H_

#include <functional>

#include "tensorflow/contrib/seastar/seastar_tensor_coding.h"
#include "tensorflow/contrib/seastar/seastar_worker_service_method.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "third_party/seastar/core/channel.hh"
#include "third_party/seastar/core/packet_queue.hh"
#include "third_party/seastar/core/temporary_buffer.hh"

namespace tensorflow {

// Required for break circular dependency
class SeastarWorkerService;

class SeastarServerTag {
 public:
  // Server Header 32B:
  // |  BBBB:4B | method:4B  |
  // |        tag:8B         |
  // | status:4B|user_data:4B|
  // |     payload_len:8B    |
  static const size_t kMethodIndex = 4;
  static const size_t kTagIndex = 8;
  static const size_t kStatusIndex = 16;
  static const size_t kUserDataIndex = 20;
  static const size_t kPayloadLenIndex = 24;
  static const size_t kHeaderSize = 32;

  SeastarServerTag(seastar::channel* seastar_channel,
                   SeastarWorkerService* seastar_worker_service);

  virtual ~SeastarServerTag();

  // Called by seastar engine, call the handler.
  void RecvReqDone(Status s);

  // Called by seastar engine.
  void SendRespDone();

  void StartResp();

  void ProcessDone(Status s);

  uint64_t GetRequestBodySize();

  char* GetRequestBodyBuffer();

  void InitResponseTensorBufs(int resp_tensor_count);
  uint64_t GetRequestMessageSize(int idx);
  char* GetRequestMessageBuffer(int idx);
  uint64_t GetRequestTensorSize(int idx);
  char* GetRequestTensorBuffer(int idx);
  int GetReqTensorCount();
  Status ParseMessage(int idx, const char* tensor_msg, size_t len);
  Status ParseTensor();
  void FillRespBody();

  bool IsRecvTensor();

  Status ParseMetaData(const char*, size_t);

 private:
  seastar::user_packet* ToUserPacket();
  seastar::user_packet* ToUserPacketWithTensor();

 public:
  seastar::channel* seastar_channel_;
  SeastarWorkerServiceMethod method_;
  uint64_t client_tag_id_;
  int32 status_;
  int req_tensor_count_;
  int resp_tensor_count_;

  SeastarBuf req_body_buf_;
  SeastarBuf resp_header_buf_;
  SeastarBuf resp_body_buf_;

  std::vector<SeastarBuf> resp_message_bufs_;
  std::vector<SeastarBuf> resp_tensor_bufs_;

  std::vector<SeastarBuf> req_tensor_bufs_;

  ParseMetaDataCallback parse_meta_data_;
  ParseMessageCallback parse_message_;
  ParseTensorCallback parse_tensor_;
  StatusCallback send_resp_;
  StatusCallback clear_;

  SeastarWorkerService* seastar_worker_service_;
};

void InitSeastarServerTag(protobuf::Message* request,
                          protobuf::Message* response, SeastarServerTag* tag);

void InitSeastarServerTag(protobuf::Message* request,
                          SeastarTensorResponse* response,
                          SeastarServerTag* tag, StatusCallback clear);

void InitSeastarServerTag(protobuf::Message* request,
                          SeastarFuseTensorResponse* response,
                          SeastarServerTag* tag,
                          StatusCallback clear);

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_SERVER_TAG_H_

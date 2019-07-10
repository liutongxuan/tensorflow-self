#include "tensorflow/contrib/seastar/seastar_remote_worker.h"

#include <utility>

#include "tensorflow/contrib/seastar/seastar_client_tag.h"
#include "tensorflow/contrib/seastar/seastar_tensor_coding.h"
#include "tensorflow/contrib/seastar/seastar_worker_interface.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

class SeastarRemoteWorker : public WorkerInterface,
                            public SeastarWorkerInterface {
public:
  explicit SeastarRemoteWorker(seastar::channel* chan,
                               WorkerCacheLogger* logger, WorkerEnv* env)
      : seastar_channel_(chan), logger_(logger), env_(env) {}

  ~SeastarRemoteWorker() override = default;

  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response,
                      StatusCallback done) override {
    GetStatusAsyncWithOptions(request, response, done, nullptr);
  }

  void GetStatusAsyncWithOptions(const GetStatusRequest* request,
                                 GetStatusResponse* response,
                                 const StatusCallback& done, CallOptions* call_opts) {
    env_->compute_pool->Schedule([this, request, response, call_opts, done]() {
      IssueRequest(request, response, SeastarWorkerServiceMethod::kGetStatus,
                   done, call_opts);
    });
  }

  void CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request,
                                CreateWorkerSessionResponse* response,
                                StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request, response,
                   SeastarWorkerServiceMethod::kCreateWorkerSession, done);
    });
  }

  void DeleteWorkerSessionAsync(CallOptions* call_opts,
                                const DeleteWorkerSessionRequest* request,
                                DeleteWorkerSessionResponse* response,
                                StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done, call_opts] {
      IssueRequest(request, response,
                   SeastarWorkerServiceMethod::kDeleteWorkerSession,
                   done, call_opts);
    });
  }

  void RegisterGraphAsync(const RegisterGraphRequest* request,
                          RegisterGraphResponse* response,
                          StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request, response,
                   SeastarWorkerServiceMethod::kRegisterGraph, done);
    });
  }

  void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                            DeregisterGraphResponse* response,
                            StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request, response,
                   SeastarWorkerServiceMethod::kDeregisterGraph, done);
    });
  }

  void RunGraphAsync(CallOptions* call_opts, const RunGraphRequest* request,
                     RunGraphResponse* response, StatusCallback done) override {
    TRACEPRINTF("Seastar RunGraph: %lld", request->step_id());
    env_->compute_pool->Schedule([this, request, response, call_opts, done]() {
      IssueRequest(request, response, SeastarWorkerServiceMethod::kRunGraph,
                   done, call_opts);
    });
  }

  void RunGraphAsync(CallOptions* call_opts, RunGraphRequestWrapper* request,
                     MutableRunGraphResponseWrapper* response,
                     StatusCallback done) override {
    TRACEPRINTF("wrapped Seastar RunGraph: %lld", request->step_id());
    env_->compute_pool->Schedule([this, request, response, call_opts, done]() {
      IssueRequest(&request->ToProto(), get_proto_from_wrapper(response),
                   SeastarWorkerServiceMethod::kRunGraph, done, call_opts);
    });
  }

  void CleanupGraphAsync(const CleanupGraphRequest* request,
                         CleanupGraphResponse* response,
                         StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request, response,
                   SeastarWorkerServiceMethod::kCleanupGraph, done);
    });
  }

  void CleanupAllAsync(const CleanupAllRequest* request,
                       CleanupAllResponse* response,
                       StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request, response,
                   SeastarWorkerServiceMethod::kCleanupAll, done);
    });
  }

  void RecvTensorAsync(CallOptions* call_opts, const RecvTensorRequest* request,
                       TensorResponse* response, StatusCallback done) override {
    done(errors::Unimplemented("SeastarWorker::RecvTensorAsync()"));
  }

  void RecvTensorAsync(CallOptions* call_opts, const RecvTensorRequest* request,
                       SeastarTensorResponse* response,
                       StatusCallback done) override {
    VLOG(1) << "RecvTensorAsync req: " << request->DebugString();
    int64 start_usec = Env::Default()->NowMicros();
    // Type-specialized logging for this method.
    bool logging_active = logger_->LoggingActive() || VLOG_IS_ON(2);

    // Don't propagate dma_ok over Seastar.
//    RecvTensorRequest* req_copy = nullptr;
//    if (request->dma_ok()) {
//      req_copy = new RecvTensorRequest;
//      *req_copy = *request;
//      req_copy->set_dma_ok(false);
//    }

    StatusCallback wrapper_done;
    const StatusCallback* cb_to_use;
    if (!logging_active) {
      cb_to_use = &done;  // No additional work to do, so just use done directly
    } else {
      wrapper_done = [this, request, response, done, start_usec](const Status& s) {
        if (logger_->LoggingActive()) {
          int64 end_usec = Env::Default()->NowMicros();
          int64 step_id = request->step_id();
          int64 bytes = response->GetTensor().TotalBytes();
          int64 send_start_usec = start_usec;
          // If a send start time was reported by the other side, use
          // that instead.  Maybe we should mark the display if we're using
          // our local time instead of the remote start time?
          if (response->send_start_micros()) {
            // send_start_micros is the timestamp taken when the
            // remote machine began to send the RecvTensor response.
            // Due to clock skew between source and dest machines, it
            // is possible that send_start_micros can be larger than
            // end_usec or less than start_usec.
            //
            // To respect causality, we enforce the invariants that
            // the RecvTensor response can not have been sent before
            // the RecvTensor request, and must have been sent before
            // it was received.
            send_start_usec = std::max(
                start_usec,
                static_cast<int64>(response->send_start_micros()));
            send_start_usec = std::min(send_start_usec, end_usec - 1);
          }
          const string& key = request->rendezvous_key();
          std::vector<string> key_parts = str_util::Split(key, ';');
          if (key_parts.size() != 5) {
            LOG(WARNING) << "Bad key: " << key;
          } else {
            logger_->RecordRecvTensor(step_id, send_start_usec, end_usec,
                                      key_parts[3],  // tensor name
                                      key_parts[0],  // src_device
                                      key_parts[2],  // dst_device
                                      bytes);
          }
        }
        done(s);
      };
//      wrapper_done = [req_copy, done](Status s) {
//        delete req_copy;
//        done(s);
//      };
      cb_to_use = &wrapper_done;
    }

    IssueRequest(request, response, SeastarWorkerServiceMethod::kRecvTensor,
                 *cb_to_use, call_opts);
  }

  void FuseRecvTensorAsync(CallOptions *call_opts,
                           const FuseRecvTensorRequest *request,
                           SeastarFuseTensorResponse *response,
                           StatusCallback done) override {
    VLOG(1) << "FuseRecvTensorAsync req: " << request->DebugString();
    int64 start_usec = Env::Default()->NowMicros();
    // Type-specialized logging for this method.
    bool logging_active = logger_->LoggingActive() || VLOG_IS_ON(2);

    StatusCallback wrapper_done;
    const StatusCallback* cb_to_use;
    if (!logging_active) {
      cb_to_use = &done;  // No additional work to do, so just use done directly
    } else {
      wrapper_done = [this, request, response, done, start_usec](const Status& s) {
        if (logger_->LoggingActive()) {
          int64 end_usec = Env::Default()->NowMicros();
          int64 step_id = request->step_id();
          int64 bytes = response->GetTotalBytes();
          int64 send_start_usec = start_usec;

          // check Recv for logging detail
          if (response->send_start_micros()) {

            send_start_usec = std::max(
                start_usec,
                static_cast<int64>(response->send_start_micros()));
            send_start_usec = std::min(send_start_usec, end_usec - 1);
          }
          const string& key = request->rendezvous_key(0);
          std::vector<string> key_parts = str_util::Split(key, ';');
          if (key_parts.size() != 5) {
            LOG(WARNING) << "Bad key: " << key;
          } else {
            logger_->RecordFuseRecvTensor(step_id, send_start_usec, end_usec,
                                      key_parts[3],  // tensor name
                                      key_parts[0],  // src_device
                                      key_parts[2],  // dst_device
                                      bytes,
                                      request->rendezvous_key_size());
          }
        }
        done(s);
      };

      cb_to_use = &wrapper_done;
    }

    IssueRequest(request, response,
                 SeastarWorkerServiceMethod::kFuseRecvTensor,
                 *cb_to_use, call_opts);
  }

  void LoggingAsync(const LoggingRequest* request, LoggingResponse* response,
                    StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request, response, SeastarWorkerServiceMethod::kLogging,
                   done);
    });
  }

  void TracingAsync(const TracingRequest* request, TracingResponse* response,
                    StatusCallback done) override {
    env_->compute_pool->Schedule([this, request, response, done]() {
      IssueRequest(request, response, SeastarWorkerServiceMethod::kTracing,
                   done);
    });
  }

  void RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                    RecvBufResponse* response, StatusCallback done) override {
    done(errors::Unimplemented("SeastarRemoteWorker::RecvBufAsync()"));
  }

  void CompleteGroupAsync(CallOptions* opts,
                          const CompleteGroupRequest* request,
                          CompleteGroupResponse* response,
                          StatusCallback done) override {
    done(errors::Unimplemented("SeastarRemoteWorker::CompleteGroupAsync()"));
  }

  void CompleteInstanceAsync(CallOptions* ops,
                             const CompleteInstanceRequest* request,
                             CompleteInstanceResponse* response,
                             StatusCallback done) override {
    done(errors::Unimplemented("SeastarRemoteWorker::CompleteInstanceAsync()"));
  }

  void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                            GetStepSequenceResponse* response,
                            StatusCallback done) override {
    done(errors::Unimplemented("SeastarRemoteWorker::GetStepSequenceAsync()"));
  }

private:
  void IssueRequest(const protobuf::Message* request,
                    protobuf::Message* response,
                    const SeastarWorkerServiceMethod method,
                    StatusCallback done, CallOptions* call_opts = nullptr) {
    auto tag = new SeastarClientTag(method, env_);
    InitSeastarClientTag(const_cast<protobuf::Message*>(request), response,
                         std::move(done), tag, call_opts);
    tag->StartReq(seastar_channel_);
  }

  void IssueRequest(const protobuf::Message* request,
                    SeastarTensorResponse* response,
                    const SeastarWorkerServiceMethod method,
                    StatusCallback done, CallOptions* call_opts = nullptr) {
    auto tag = new SeastarClientTag(method, env_);
    InitSeastarClientTag(const_cast<protobuf::Message*>(request), response,
                         std::move(done), tag, call_opts);
    tag->StartReq(seastar_channel_);
  }

  void IssueRequest(const protobuf::Message *request,
                    SeastarFuseTensorResponse *response,
                    const SeastarWorkerServiceMethod method,
                    StatusCallback done,
                    CallOptions *call_opts = nullptr) {
    auto tag = new SeastarClientTag(method, env_, response->GetFuseCount());
    tag->InitTensorBuffers(tag->req_tensor_count_);
    InitSeastarClientTag(const_cast<protobuf::Message *>(request),
                         response, std::move(done), tag, call_opts);
    tag->StartReq(seastar_channel_);
  }

private:
  seastar::channel* seastar_channel_;
  WorkerCacheLogger* logger_;
  WorkerEnv* env_;

  TF_DISALLOW_COPY_AND_ASSIGN(SeastarRemoteWorker);
};

WorkerInterface* NewSeastarRemoteWorker(seastar::channel* seastar_channel,
                                        WorkerCacheLogger* logger,
                                        WorkerEnv* env) {
  return new SeastarRemoteWorker(seastar_channel, logger, env);
}

}  // namespace tensorflow

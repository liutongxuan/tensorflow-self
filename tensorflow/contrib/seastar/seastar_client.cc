#include "tensorflow/contrib/seastar/seastar_client.h"
#include "tensorflow/contrib/seastar/seastar_client_tag.h"
#include "tensorflow/core/platform/logging.h"
#include "third_party/seastar/core/reactor.hh"
#include "third_party/seastar/core/sleep.hh"
#include "tensorflow/contrib/seastar/seastar_message.h"

namespace tensorflow {

SeastarClient::Connection::Connection(seastar::connected_socket&& fd,
                                      seastar::channel* chan,
                                      SeastarTagFactory* tag_factory,
                                      seastar::socket_address addr)
    : channel_(chan), tag_factory_(tag_factory), addr_(addr) {
  fd_ = std::move(fd);
  fd_.set_nodelay(true);
  read_buf_ = fd_.input();
  channel_->init(seastar::engine().get_packet_queue(), std::move(fd_.output()));
}

SeastarClient::Connection::~Connection() {
  fd_.input().close();
  fd_.shutdown_input();
  fd_.shutdown_output();
}

seastar::future<> SeastarClient::Connection::Read() {
  return read_buf_.read_exactly(SeastarClientTag::HEADER_SIZE)
      .then([this](auto&& header) {
        if (header.size() != SeastarClientTag::HEADER_SIZE) {
          return seastar::make_ready_future();
        }
        if (memcmp(header.get(), SeastarServerTag::HEADER_SIGN, 8) != 0) {
          string out_header(header.get(), header.get() + 8);
          LOG(ERROR) << "Seastar client read header error: " << out_header;
          return seastar::make_ready_future();
        }
        auto tag = tag_factory_->CreateSeastarClientTag(header);
        if (tag->status_ != 0) {
          return read_buf_.read_exactly(tag->resp_err_msg_len_)
              .then([this, tag](auto&& err_msg) {
                std::string msg =
                    std::string(err_msg.get(), tag->resp_err_msg_len_);
                if (tag->resp_err_msg_len_ == 0) {
                  msg = "Empty error msg.";
                }
                tag->RecvRespDone(
                    Status(static_cast<error::Code>(tag->status_), msg));
                return seastar::make_ready_future();
              });
        }

        if (tag->IsRecvTensor()) {
          // handle tensor response
          auto message_size = tag->GetResponseMessageSize();
          auto message_buffer = tag->GetResponseMessageBuffer();
          return read_buf_.read_exactly(message_size)
              .then([this, tag, message_size, message_buffer](auto&& message) {
                memcpy(message_buffer, message.get(), message.size());
                auto sts = tag->ParseMessage();
                auto tensor_size = tag->GetResponseTensorSize();
                auto tensor_buffer = tag->GetResponseTensorBuffer();
                if (tensor_size == 0) {
                  tag->RecvRespDone(tensorflow::Status());
                  return seastar::make_ready_future();
                }
                return read_buf_.read_exactly(tensor_size)
                    .then(
                        [this, tag, tensor_size, tensor_buffer](auto&& tensor) {
                          if (tensor.size() != tensor_size) {
                            LOG(WARNING)
                                << "Expected read size is:" << tensor_size
                                << ", but real tensor size:" << tensor.size();
                            tag->RecvRespDone(Status(
                                error::UNKNOWN,
                                "Seastar Client: read invalid tensorbuf"));
                            return seastar::make_ready_future();
                          }
                          //address of tensor_buffer is returned by
                          //tensorflow allocator [zero-copy]
                          memcpy(tensor_buffer, tensor.get(), tensor.size());
                          tag->RecvRespDone(tensorflow::Status());
                          return seastar::make_ready_future();
                        });
              });
        } else if (tag->IsFuseRecvTensor()) {

          // handle tensor response & fuse recv tensor response
          int *recv_count = new int(tag->req_tensor_count_);
          int *idx = new int(0);
          bool *error = new bool(false);
          return this->ReapeatReadTensors(tag, recv_count, idx, error);
        } else {
          // handle general response
          auto resp_body_size = tag->GetResponseBodySize();
          if (resp_body_size == 0) {
            tag->RecvRespDone(tensorflow::Status());
            return seastar::make_ready_future();
          }

          auto resp_body_buffer = tag->GetResponseBodyBuffer();
          return read_buf_.read_exactly(resp_body_size)
              .then([this, tag, resp_body_size, resp_body_buffer](auto&& body) {
                if (body.size() != resp_body_size) {
                  LOG(WARNING) << "Expected read size is:" << resp_body_size
                               << ", but real size is:" << body.size();
                  tag->RecvRespDone(tensorflow::Status(
                      error::UNKNOWN, "Seastar Client: read invalid msgbuf"));
                  return seastar::make_ready_future();
                }
                memcpy(resp_body_buffer, body.get(), body.size());
                tag->RecvRespDone(tensorflow::Status());
                return seastar::make_ready_future();
              });
        }
      });
}

seastar::future<> SeastarClient::Connection::ReapeatReadTensors(SeastarClientTag* tag,
                                                                int* count,
                                                                int* idx,
                                                                bool* error) {
  return seastar::do_until(
      [this, tag, count, idx, error] {
        if (*error || *idx == *count) {
          delete count;
          delete idx;
          // NOTE(rangeng.llb): If error happens, tag->RecvRespDone has been called.
          if (!(*error)) {
            //switch to another thread
            tag->ScheduleProcess([tag] {
              tag->HandleResponse(tensorflow::Status());
            });
          }
          delete error;
          return true;
        } else {
          return false;
        }
      },
      [this, tag, idx, error] {
        return read_buf_.read_exactly(SeastarMessage::kMessageTotalBytes)
            .then([this, tag, idx, error] (auto&& tensor_msg) {
              CHECK_CONNECTION_CLOSE(tensor_msg.size());
              auto sts = tag->ParseFuseMessage(*idx,
                                               tensor_msg.get(),
                                               tensor_msg.size());
              auto tensor_size = tag->GetResponseTensorSize(*idx);
              auto tensor_buffer = tag->GetResponseTensorBuffer(*idx);
              ++(*idx);

              if (tensor_size == 0) {
                return seastar::make_ready_future();
              }
              if (tensor_size >= SeastarClient::_8KB) {
                return read_buf_.read_exactly(tensor_buffer, tensor_size)
                    .then([this, tag, idx, error, tensor_size, tensor_buffer] (auto read_size) {
                      CHECK_CONNECTION_CLOSE(read_size);
                      if (read_size != tensor_size) {
                        LOG(WARNING) << "warning expected read size is:" << tensor_size
                                     << ", actual read tensor size:" << read_size;
                        tag->ScheduleProcess([tag] {
                          tag->HandleResponse(tensorflow::Status(error::UNKNOWN,
                                                                 "Seastar Client: read invalid tensorbuf"));
                        });
                        *error = true;
                        return seastar::make_ready_future();
                      }
                      // No need copy here
                      return seastar::make_ready_future();
                    });
              } else {
                return read_buf_.read_exactly(tensor_size)
                    .then([this, tag, idx, error, tensor_size, tensor_buffer] (auto&& tensor) {
                      CHECK_CONNECTION_CLOSE(tensor.size());
                      if (tensor.size() != tensor_size) {
                        LOG(WARNING) << "warning expected read size is:" << tensor_size
                                     << ", actual read tensor size:" << tensor.size();
                        tag->ScheduleProcess([tag] {
                          tag->HandleResponse(tensorflow::Status(error::UNKNOWN,
                                                                 "Seastar Client: read invalid tensorbuf"));
                        });
                        *error = true;
                        return seastar::make_ready_future();
                      }
                      memcpy(tensor_buffer, tensor.get(), tensor.size());
                      return seastar::make_ready_future();
                    });
              }
            });
      });
}

void SeastarClient::Connect(seastar::ipv4_addr seastar_addr,
                            const std::string& addr,
                            seastar::channel* chan,
                            SeastarTagFactory* tag_factory) {
  seastar::socket_address local =
      seastar::socket_address(::sockaddr_in{AF_INET, INADDR_ANY, 0});

  using namespace std::chrono_literals;
  seastar::with_timeout(seastar::lowres_clock::now() + 60s,
                        seastar::engine().net()
                            .connect(seastar::make_ipv4_address(seastar_addr),
                                     local, seastar::transport::TCP))
//  seastar::engine().net()
//      .connect(seastar::make_ipv4_address(seastar_addr),
//               local, seastar::transport::TCP)
      .then(
          [this, chan, addr, seastar_addr, tag_factory](
              seastar::connected_socket fd) {
            auto conn = new Connection(std::move(fd), chan, tag_factory,
                                       seastar::socket_address(seastar_addr));
            seastar::do_until([conn] { return conn->read_buf_.eof(); },
                              [conn] { return conn->Read(); })
                .then_wrapped(
                    [this, conn, seastar_addr, addr, chan, tag_factory](auto&& f) {
                      try {
                        f.get();
                        LOG(INFO) << "Remote server closed connection: addr = "
                                  << addr;
                      } catch (std::exception& ex) {
                        LOG(WARNING) << "Read got an exception: " << ex.what()
                                     << ", errno = " << errno << ", addr = " << addr;
                      }
                      chan->set_channel_broken();
                      delete conn;
//                      auto sec = std::chrono::seconds(retry_sec);
//                      LOG(INFO) << "Reconnect to " << addr << " after " << retry_sec << "s";
//                      seastar::sleep(sec).then(
//                          [this, seastar_addr, addr, chan, tag_factory, retry_sec] {
//                            this->Connect(seastar_addr, addr, chan, tag_factory, retry_sec * 2);
//                          });
                    });
            return seastar::make_ready_future();
          })
      .handle_exception(
          [this, chan, seastar_addr, addr, tag_factory](auto ep) {
            LOG(WARNING) << "seastar client exception: " << ep;
            using namespace std::chrono_literals;
            return seastar::sleep(3s).then(
                [this, chan, seastar_addr, addr, tag_factory] {
                  LOG(INFO) << "Reconnect to server " << addr;
                  this->Connect(seastar_addr, addr, chan, tag_factory);
                });
          });
}

}  // namespace tensorflow

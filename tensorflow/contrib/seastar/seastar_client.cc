#include "tensorflow/contrib/seastar/seastar_client.h"
#include "tensorflow/contrib/seastar/seastar_client_tag.h"
#include "tensorflow/core/platform/logging.h"
#include "third_party/seastar/core/reactor.hh"
#include "third_party/seastar/core/sleep.hh"

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

seastar::future<> SeastarClient::Connection::Read() {
  return read_buf_.read_exactly(SeastarClientTag::HEADER_SIZE)
      .then([this](auto&& header) {
        if (header.size() == 0) {
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
          // handle tensor response & fuse recv tensor response
          int *recv_count = new int(tag->resp_tensor_count_);
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
        return _read_buf.read_exactly(StarMessage::kMessageTotalBytes)
          .then([this, tag, idx, error] (auto&& tensor_msg) {
            CHECK_CONNECTION_CLOSE(tensor_msg.size());
            tag->ParseTensorMessage(*idx, tensor_msg.get(), tensor_msg.size());
            auto tensor_size = tag->GetResponseTensorSize(*idx);
            auto tensor_buffer = tag->GetResponseTensorBuffer(*idx);
            ++(*idx);

            if (tensor_size == 0) {
              return seastar::make_ready_future();
            }
            if (tensor_size >= _8KB) {
              return _read_buf.read_exactly(tensor_buffer, tensor_size)
                .then([this, tag, error, tensor_size, tensor_buffer] (auto read_size) {
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
              return _read_buf.read_exactly(tensor_size)
                .then([this, tag, error, tensor_size, tensor_buffer] (auto&& tensor) {
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

void SeastarClient::Connect(seastar::ipv4_addr server_addr, std::string s,
                            seastar::channel* chan,
                            SeastarTagFactory* tag_factory) {
  seastar::socket_address local =
      seastar::socket_address(::sockaddr_in{AF_INET, INADDR_ANY, {0}});

  seastar::engine()
      .net()
      .connect(seastar::make_ipv4_address(server_addr), local,
               seastar::transport::TCP)
      .then([this, chan, s, server_addr,
             tag_factory](seastar::connected_socket fd) {
        auto conn = new Connection(std::move(fd), chan, tag_factory,
                                   seastar::socket_address(server_addr));

        seastar::do_until([conn] { return conn->read_buf_.eof(); },
                          [conn] { return conn->Read(); })
            .then_wrapped([this, conn, s, chan](auto&& f) {
              try {
                f.get();
                VLOG(2) << "Remote closed the connection: addr = " << s;
              } catch (std::exception& ex) {
                LOG(WARNING) << "Read got an exception: " << errno
                             << ", addr = " << s;
              }
              chan->set_channel_broken();
            });
        return seastar::make_ready_future();
      })
      .handle_exception([this, chan, server_addr, s, tag_factory](auto ep) {
        using namespace std::chrono_literals;
        return seastar::sleep(1s).then(
            [this, chan, server_addr, s, tag_factory] {
              this->Connect(server_addr, s, chan, tag_factory);
            });
      });
}

}  // namespace tensorflow

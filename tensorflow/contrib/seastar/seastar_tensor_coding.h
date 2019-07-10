#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_TENSOR_CODING_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_TENSOR_CODING_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

struct SeastarBuf {
  uint64_t len_ = 0;
  char* data_ = nullptr;
  bool owned_ = true; //when false, tf will take care of deleting
  ~SeastarBuf(){
    if(owned_){
      delete[] data_;
    }
  }
};

class SeastarTensorResponse {
 public:
  virtual ~SeastarTensorResponse() = default;

  void SetIsDead(bool is_dead) { is_dead_ = is_dead; }

  bool GetIsDead() const { return is_dead_; }

  // for dst device
  void InitAlloc(Device* d, const AllocatorAttributes& aa);

  Allocator* GetAlloc() { return allocator_; }

  AllocatorAttributes GetAllocAttributes() { return alloc_attrs_; }

  Device* GetDevice() const { return device_; }

  bool GetOnHost() const { return on_host_; }

  void SetTensor(const Tensor& tensor) { tensor_ = tensor; }

  Tensor& GetTensor() { return tensor_; }

  TensorProto& GetTensorProto() { return tensor_proto_; }

  void Clear();

  void SetDataType(DataType data_type) { data_type_ = data_type; }

  DataType GetDataType() { return data_type_; }

  inline void set_send_start_micros(int64 value) { send_start_micros_ = value; }

  inline int64 send_start_micros() { return send_start_micros_; }

 private:
  bool is_dead_ = false;
  bool on_host_ = false;

  // for dst device
  Device* device_ = nullptr;
  AllocatorAttributes alloc_attrs_;
  Allocator* allocator_ = nullptr;

  Tensor tensor_;
  TensorProto tensor_proto_;
  DataType data_type_;
  int64 send_start_micros_;
};

class SeastarFuseTensorResponse : public SeastarTensorResponse {
 public:
  SeastarFuseTensorResponse() : fuse_count_(0), initialized_(false) {}
  virtual ~SeastarFuseTensorResponse() {}

  void Init(int fuse_count)  {
    fuse_count_ = fuse_count;
    tensors_.resize(fuse_count);
    tensor_protos_.resize(fuse_count);
    data_types_.resize(fuse_count);
    is_deads_.resize(fuse_count);
    initialized_ = true;
  }

  int GetFuseCount() { return fuse_count_; }
  bool Initialized() {return initialized_;}

  void SetIsDeadByIndex(int idx, bool is_dead) { is_deads_[idx] = is_dead; }
  bool GetIsDeadByIndex(int idx) const { return is_deads_[idx]; }
  const std::vector<bool>& GetIsDeads() const { return is_deads_; }

  void SetTensorByIndex(int idx, const Tensor& tensor) { tensors_[idx] = tensor; }
  const Tensor& GetTensorByIndex(int idx) const { return tensors_[idx]; }
  const std::vector<Tensor>& GetTensors() const { return tensors_; }
  TensorProto& GetTensorProtoByIndex(int idx) { return tensor_protos_[idx]; }

  void Clear();

  void SetDataTypeByIndex(int idx, DataType data_type) { data_types_[idx] = data_type; }
  DataType GetDataTypeByIndex(int idx) { return data_types_[idx]; }
  size_t GetTotalBytes(){
    size_t total_bytes = 0;
    for (const auto &each : tensors_){
      total_bytes += each.TotalBytes();
    }
    return total_bytes;
  }

 private:
  bool initialized_;
  int fuse_count_;
  std::vector<Tensor> tensors_;
  std::vector<TensorProto> tensor_protos_;
  std::vector<DataType> data_types_;
  std::vector<bool> is_deads_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_TENSOR_CODING_H_

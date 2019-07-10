// Copyright (c) 2018, Alibaba Inc.
// All right reserved.
//
// Author: Liangbin LI <rangeng.llb@taobao.com>
// Created: 2018/02/07
// Description:

#ifndef TENSORFLOW_KERNELS_FUSERECV_OPS_H_
#define TENSORFLOW_KERNELS_FUSERECV_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class FuseRecvOp : public AsyncOpKernel {
 public:
  explicit FuseRecvOp(OpKernelConstruction* ctx);
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  std::vector<string> key_prefixs_;
  std::vector<Rendezvous::ParsedKey> parsed_keys_;
  bool hostmem_sendrecv_;
  int fuse_count_;

  TF_DISALLOW_COPY_AND_ASSIGN(FuseRecvOp);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_FUSERECV_OPS_H_

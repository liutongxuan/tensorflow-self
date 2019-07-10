// Copyright (c) 2018, Alibaba Inc.
// All right reserved.
//
// Author: Liangbin LI <rangeng.llb@taobao.com>
// Created: 2018/02/07
// Description:

#ifndef TENSORFLOW_TENSORFLOW_CORE_KERNELS_FUSERECV_OP_H_
#define TENSORFLOW_TENSORFLOW_CORE_KERNELS_FUSERECV_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/logging.h"

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

}

#endif //TENSORFLOW_TENSORFLOW_CORE_KERNELS_FUSERECV_OP_H_

#ifndef _BROADCAST_EXPAND_H_
#define _BROADCAST_EXPAND_H_

#include <torch/script.h>
namespace mmdeploy {
namespace torch_jit {
using torch::jit::Graph;

void BroadcastExpand(const std::shared_ptr<Graph>& graph);
}  // namespace torch_jit
}  // namespace mmdeploy

#endif

#include "broadcast_expand.h"

#include <ATen/ATen.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

#include <unordered_set>

namespace mmdeploy {
namespace torch_jit {

using torch::jit::Block;
using torch::jit::IValue;
using torch::jit::Node;
using torch::jit::Symbol;
using torch::jit::TensorType;
using torch::jit::use_list;
using torch::jit::Value;

namespace prim {
using namespace ::c10::prim;
}

namespace aten {
using namespace ::c10::aten;
}

static bool checkBroadcastExpand(const use_list& uses, bool& need_reshape) {
  // check if all users can be broadcasted.
  const static std::unordered_set<Symbol> broadcast_user{aten::add, aten::sub, aten::mul,
                                                         aten::div};

  for (auto use : uses) {
    auto user = use.user;

    if (user->kind() == aten::to) {
      // if node is `aten::to`, loop over it's users.
      auto user_out = user->output();
      // do nothing if expand is not used.
      if (!user_out->hasUses()) return false;
      bool ret = checkBroadcastExpand(user_out->uses(), need_reshape);
      if (!ret) return false;
    } else if (user->kind() == prim::ListConstruct) {
      // if user is list and all use of this list is aten::index
      // it can be broadcasted.
      auto list_out = user->output();
      if (!list_out->hasUses()) return false;
      for (auto list_use : list_out->uses()) {
        if (!list_use.user->kind() == aten::index) {
          return false;
        }
        need_reshape = true;
      }

    } else if (broadcast_user.count(user->kind()) != 0) {
      // If user is binary ops it can be broadcasted.
      continue;
    } else {
      // Any other kind is invalid.
      return false;
    }
  }

  return true;
}

//   printf("output->type()->cast<TensorType>()->dim(): %d\n",
//          output->type()->cast<TensorType>()->dim().value());

void BroadcastExpand(Node* node) {
  auto input = node->inputs()[0];
  auto output = node->output();
  if (!output->hasUses()) return;
  auto uses = output->uses();

  // check if broadcast is necessary.
  bool need_reshape = false;
  bool need_broadcast = checkBroadcastExpand(uses, need_reshape);
  if (!need_broadcast) return;

  // replace all use
  if (need_reshape) {
    auto in_dim = input->type()->cast<TensorType>()->dim();
    auto out_dim = output->type()->cast<TensorType>()->dim();
    if (!in_dim.has_value() || !out_dim.has_value()) {
      return;
    }
    // case:
    // index = index.expand(...)
    // x = x[..., index, ...]
    // If we remove the expand, we should unsqueeze it to the same rank.
    int num_unsqueeze = out_dim.value() - in_dim.value();
    if (num_unsqueeze == 0) {
      output->replaceAllUsesWith(input);
    } else {
      auto graph = node->owningGraph();
      auto const_0 = graph->insertConstant(IValue(0));
      Value* index_input = input;
      for (int i = 0; i < num_unsqueeze; ++i) {
        auto unsqueeze_node = graph->create(aten::unsqueeze, {index_input, const_0});
        unsqueeze_node->insertBefore(node);
        index_input = unsqueeze_node->output();
      }

      output->replaceAllUsesWith(index_input);
    }
  } else {
    output->replaceAllUsesWith(input);
  }
  node->destroy();
}

void BroadcastExpand(Block* block) {
  auto graph = block->owningGraph();
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      BroadcastExpand(block);
    }

    if (node->kind() == aten::expand || node->kind() == aten::expand_as ||
        node->kind() == aten::repeat) {
      BroadcastExpand(node);
    }
  }
}

void BroadcastExpand(const std::shared_ptr<Graph>& graph) {
  BroadcastExpand(graph->block());
  torch::jit::EliminateDeadCode(graph);
}

}  // namespace torch_jit
}  // namespace mmdeploy

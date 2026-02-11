#pragma once

#include "tensorium/IR/IRBase.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tensorium::backend {

struct TensorProductIR final : ExprIR {
  std::unique_ptr<ExprIR> lhs;
  std::unique_ptr<ExprIR> rhs;
  TensorProductIR(std::unique_ptr<ExprIR> L, std::unique_ptr<ExprIR> R)
      : ExprIR(Kind::TensorProduct), lhs(std::move(L)), rhs(std::move(R)) {}
};

struct ContractionIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  std::vector<std::string> summedIndices;
  explicit ContractionIR(std::unique_ptr<ExprIR> expr)
      : ExprIR(Kind::Contraction), in(std::move(expr)) {}
};

struct IndexRenameIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  std::string from;
  std::string to;
  IndexRenameIR(std::unique_ptr<ExprIR> expr, std::string fromIndex,
                std::string toIndex)
      : ExprIR(Kind::IndexRename), in(std::move(expr)),
        from(std::move(fromIndex)), to(std::move(toIndex)) {}
};

struct IndexPermuteIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  std::vector<std::string> order;
  IndexPermuteIR(std::unique_ptr<ExprIR> expr, std::vector<std::string> outOrder)
      : ExprIR(Kind::IndexPermute), in(std::move(expr)),
        order(std::move(outOrder)) {}
};

struct TraceIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  std::vector<std::string> tracedIndices;
  explicit TraceIR(std::unique_ptr<ExprIR> expr)
      : ExprIR(Kind::Trace), in(std::move(expr)) {}
};

} // namespace tensorium::backend

#pragma once

#include "tensorium/IR/IRBase.hpp"

#include <memory>
#include <string>
#include <utility>

namespace tensorium::backend {

struct PartialDerivativeIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  std::string coordIndex;
  PartialDerivativeIR(std::unique_ptr<ExprIR> expr, std::string idx)
      : ExprIR(Kind::PartialDerivative), in(std::move(expr)),
        coordIndex(std::move(idx)) {}
};

struct GradientIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  explicit GradientIR(std::unique_ptr<ExprIR> expr)
      : ExprIR(Kind::Gradient), in(std::move(expr)) {}
};

struct CovariantDerivativeIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  std::string derivIndex;
  bool contravariant = false;
  bool hasConnectionTensor = false;
  CovariantDerivativeIR(std::unique_ptr<ExprIR> expr, std::string idx)
      : ExprIR(Kind::CovariantDerivative), in(std::move(expr)),
        derivIndex(std::move(idx)) {}
};

struct DivergenceIR final : ExprIR {
  std::unique_ptr<ExprIR> in;
  std::string contractedIndex;
  explicit DivergenceIR(std::unique_ptr<ExprIR> expr)
      : ExprIR(Kind::Divergence), in(std::move(expr)) {}
};

} // namespace tensorium::backend

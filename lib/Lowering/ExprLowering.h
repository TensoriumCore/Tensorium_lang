#pragma once

#include "tensorium/AST/IndexedAST.hpp"
#include "tensorium/IR/DomainIR.hpp"

#include <memory>

namespace tensorium::backend {

std::unique_ptr<ExprIR>
lowerIndexedExpr(const tensorium::IndexedExpr *e,
                 bool materializeImplicitContraction,
                 bool hasConnectionTensor);

} // namespace tensorium::backend

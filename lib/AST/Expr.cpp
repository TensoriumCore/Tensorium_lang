#include "tensorium/AST/AST.hpp"

namespace tensorium {
//
// NumberExpr::NumberExpr(double v) : value(v) {}
// void NumberExpr::accept(ExprVisitor &V) const { V.visit(*this); }
//
// VarExpr::VarExpr(std::string n) : name(std::move(n)) {}
// void VarExpr::accept(ExprVisitor &V) const { V.visit(*this); }
//
// BinaryExpr::BinaryExpr(std::unique_ptr<Expr> L, char Op,
//                        std::unique_ptr<Expr> R)
//     : lhs(std::move(L)), rhs(std::move(R)), op(Op) {}
// void BinaryExpr::accept(ExprVisitor &V) const { V.visit(*this); }
//
// ParenExpr::ParenExpr(std::unique_ptr<Expr> e) : inner(std::move(e)) {}
// void ParenExpr::accept(ExprVisitor &V) const { V.visit(*this); }
//
// void CallExpr::accept(ExprVisitor &V) const { V.visit(*this); }
//
// IndexedVarExpr::IndexedVarExpr(std::string b, std::vector<std::string> idx)
//     : base(std::move(b)), indices(std::move(idx)) {}
// void IndexedVarExpr::accept(ExprVisitor &V) const { V.visit(*this); }
//
} // namespace tensorium

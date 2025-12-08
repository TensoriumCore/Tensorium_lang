#pragma once

namespace tensorium {

struct NumberExpr;
struct VarExpr;
struct BinaryExpr;
struct CallExpr;
struct ParenExpr;
struct IndexedVarExpr;

// class ExprVisitor {
// public:
//   virtual ~ExprVisitor() = default;
//   virtual void visit(const NumberExpr &E) = 0;
//   virtual void visit(const VarExpr &E) = 0;
//   virtual void visit(const BinaryExpr &E) = 0;
//   virtual void visit(const CallExpr &E) = 0;
//   virtual void visit(const ParenExpr &E) = 0;
//   virtual void visit(const IndexedVarExpr &E) = 0;
// };

} // namespace tensorium

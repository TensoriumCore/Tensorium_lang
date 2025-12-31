#include "tensorium/AST/ASTPrinter.hpp"
#include "tensorium/AST/AST.hpp"
#include "tensorium/AST/Visitor.hpp"
#include <iostream>

namespace tensorium {

class ASTPrinter : public ExprVisitor {
public:
  void visit(const NumberExpr &E) override { std::cout << E.value; }

  void visit(const VarExpr &E) override { std::cout << E.name; }

  void visit(const BinaryExpr &E) override {
    std::cout << "(";
    E.lhs->accept(*this);
    std::cout << " " << E.op << " ";
    E.rhs->accept(*this);
    std::cout << ")";
  }

  void visit(const ParenExpr &E) override {
    std::cout << "(";
    E.inner->accept(*this);
    std::cout << ")";
  }

  void visit(const CallExpr &E) override {
    std::cout << E.callee << "(";
    for (size_t i = 0; i < E.args.size(); ++i) {
      E.args[i]->accept(*this);
      if (i + 1 < E.args.size())
        std::cout << ", ";
    }
    std::cout << ")";
  }

  void visit(const IndexedVarExpr &E) override {
    std::cout << E.base << "[";
    for (size_t i = 0; i < E.indices.size(); ++i) {
      std::cout << E.indices[i];
      if (i + 1 < E.indices.size())
        std::cout << ",";
    }
    std::cout << "]";
  }
};

void printExpr(const Expr *e) {
  if (!e)
    return;
  ASTPrinter P;
  e->accept(P);
}

void printProgram(const Program &prog) {
  std::cout << "=== Program AST ===\n";
  if (!prog.fields.empty()) {
    std::cout << "\nFields:\n";
    for (const auto &f : prog.fields) {
      std::cout << "  field ";

      switch (f.kind) {
      case TensorKind::Scalar:
        std::cout << "scalar ";
        break;
      case TensorKind::Vector:
        std::cout << "vector ";
        break;
      case TensorKind::Covector:
        std::cout << "covector ";
        break;
      case TensorKind::CovTensor2:
        std::cout << "cov_tensor2";
        break;
      case TensorKind::ConTensor2:
        std::cout << "con_tensor2";
        break;
	  case TensorKind::CovTensor3:
        std::cout << "cov_tensor3";
        break;
      case TensorKind::ConTensor3:
        std::cout << "con_tensor3";
        break;
      case TensorKind::ConTensor4:
        std::cout << "con_tensor4";
        break;
      case TensorKind::CovTensor4:
        std::cout << "cov_tensor4";
        break;
      case TensorKind::MixedTensor:
        std::cout << "mixed_tensor";
        break;
      }

      std::cout << f.name;

      if (!f.indices.empty()) {
        std::cout << "[";
        for (size_t i = 0; i < f.indices.size(); ++i) {
          std::cout << f.indices[i];
          if (i + 1 < f.indices.size())
            std::cout << ",";
        }
        std::cout << "]";
      }
      std::cout << "\n";
    }
  }

  if (prog.simulation) {
    const auto &sim = *prog.simulation;

    std::cout << "\nSimulation:\n";
    std::cout << "  dimension = " << sim.dimension << "\n";

    std::cout << "  resolution = [";
    for (size_t i = 0; i < sim.resolution.size(); ++i) {
      std::cout << sim.resolution[i];
      if (i + 1 < sim.resolution.size())
        std::cout << ",";
    }
    std::cout << "]\n";

    std::cout << "  time:\n";
    std::cout << "    dt = " << sim.time.dt << "\n";

    std::cout << "  spatial:\n";
    std::cout << "    order = " << sim.spatial.order << "\n";
  }
  for (const auto &evo : prog.evolutions) {
    std::cout << "Evolution " << evo.name << " {\n";
    for (const auto &eq : evo.equations) {
      std::cout << "  dt " << eq.fieldName;
      if (!eq.indices.empty()) {
        std::cout << "[";
        for (size_t i = 0; i < eq.indices.size(); ++i) {
          std::cout << eq.indices[i];
          if (i + 1 < eq.indices.size())
            std::cout << ",";
        }
        std::cout << "]";
      }
      std::cout << " = ";
      printExpr(eq.rhs.get());
      std::cout << "\n";
    }
    std::cout << "}\n";
  }
}

} // namespace tensorium

#include "tensorium/AST/ASTPrinter.hpp"
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
	for(size_t i=0; i<E.args.size(); ++i) {
		E.args[i]->accept(*this);
		if(i+1 < E.args.size()) std::cout << ", ";
	}
	std::cout << ")";
  }
  
  void visit(const IndexedVarExpr &E) override {
	std::cout << E.base << "[";
	for(size_t i=0; i<E.indices.size(); ++i) {
		std::cout << E.indices[i];
		if(i+1 < E.indices.size()) std::cout << ",";
	}
	std::cout << "]";
  }
};

void printExpr(const Expr *e) {
	if(!e) return;
	ASTPrinter P;
	e->accept(P);
}

void printProgram(const Program &prog) {
	std::cout << "=== Program AST ===\n";
	for(const auto &evo : prog.evolutions) {
		std::cout << "Evolution " << evo.name << " {\n";
		for(const auto &eq : evo.equations) {
			std::cout << "  dt " << eq.fieldName;
			if(!eq.indices.empty()) {
				std::cout << "[";
				for(auto &idx : eq.indices) std::cout << idx << ","; // simplifié
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

#pragma once
#include "tensorium/AST/AST.hpp"

namespace tensorium {
    void printProgram(const Program &prog);
    void printExpr(const Expr *e);
}

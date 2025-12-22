
#include "tensorium/Runtime/Eval.hpp"

namespace tensorium::runtime {

static double evalVar(const backend::VarIR* v, const ScalarEnv& env) {
  using backend::VarKind;

  if (v->vkind == VarKind::Field) {
    auto it = env.fieldPtr.find(v->name);
    if (it == env.fieldPtr.end() || !it->second)
      throw std::runtime_error("runtime: missing field value for '" + v->name + "'");
    return *(it->second);
  }

  if (v->vkind == VarKind::Param) {
    auto it = env.params.find(v->name);
    if (it == env.params.end())
      throw std::runtime_error("runtime: missing param '" + v->name + "'");
    return it->second;
  }

  throw std::runtime_error("runtime: unsupported var kind for '" + v->name + "'");
}

double evalScalar(const backend::ExprIR* e, const ScalarEnv& env) {
  using backend::ExprIR;

  if (!e) throw std::runtime_error("runtime: null expr");

  switch (e->kind) {
  case ExprIR::Kind::Number: {
    auto* n = static_cast<const backend::NumberIR*>(e);
    return n->value;
  }
  case ExprIR::Kind::Var: {
    auto* v = static_cast<const backend::VarIR*>(e);
    return evalVar(v, env);
  }
  case ExprIR::Kind::Binary: {
    auto* b = static_cast<const backend::BinaryIR*>(e);
    const double L = evalScalar(b->lhs.get(), env);
    const double R = evalScalar(b->rhs.get(), env);

    const std::string& op = b->op;
    if (op == "+") return L + R;
    if (op == "-") return L - R;
    if (op == "*") return L * R;
    if (op == "/") return L / R;

    throw std::runtime_error("runtime: unsupported binary op '" + op + "'");
  }
  case ExprIR::Kind::Call:
    throw std::runtime_error("runtime: calls not supported yet in scalar runtime");
  }

  throw std::runtime_error("runtime: unreachable");
}

} // namespace tensorium::runtime

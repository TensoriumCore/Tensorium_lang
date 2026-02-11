#include "tensorium/Backend/BackendBuilder.hpp"
#include "tensorium/Core/IndexSet.h"
#include "tensorium/Lex/Lexer.hpp"
#include "tensorium/Parse/Parser.hpp"
#include "tensorium/Sema/Sema.hpp"

#include <iostream>
#include <string>

using namespace tensorium;

static bool testConTensor3Lowering() {
  static const char *kSource = R"(
    field con_tensor3 A[i,j,k]

    simulation {
      dimension = 1
      resolution = [8]
      time { dt = 0.1 integrator = euler }
      spatial { scheme = fd derivative = centered order = 2 }
    }

    evolution E {
      dt A[i,j,k] = A[i,j,k]
    }
  )";

  Lexer lex(kSource);
  Parser parser(lex);
  Program prog = parser.parseProgram();
  SemanticAnalyzer sem(prog, CompilationMode::Executable);
  backend::ModuleIR mod = backend::BackendBuilder::build(prog, sem);

  for (const auto &field : mod.fields) {
    if (field.name != "A")
      continue;

    if (field.kind != backend::FieldKind::ConTensor3) {
      std::cerr << "FAIL: expected backend kind ConTensor3 for field A\n";
      return false;
    }
    if (field.tensorType.up != 3 || field.tensorType.down != 0) {
      std::cerr << "FAIL: expected field A variance up=3 down=0\n";
      return false;
    }
    return true;
  }

  std::cerr << "FAIL: field A not found in lowered backend module\n";
  return false;
}

static bool testIndexSetPolicy() {
  for (char idx : core::kTensorIndices) {
    if (!core::isTensorIndexChar(idx)) {
      std::cerr << "FAIL: expected accepted tensor index char '" << idx
                << "'\n";
      return false;
    }
    if (!core::isTensorIndexName(std::string(1, idx))) {
      std::cerr << "FAIL: expected accepted tensor index name '" << idx
                << "'\n";
      return false;
    }
  }

  const char rejectedChars[] = {'a', 'x', 'z', '0', 'I', '_'};
  for (char idx : rejectedChars) {
    if (core::isTensorIndexChar(idx)) {
      std::cerr << "FAIL: expected rejected tensor index char '" << idx
                << "'\n";
      return false;
    }
  }

  const std::string rejectedNames[] = {"", "ij", "p", "i0", "_", "theta"};
  for (const auto &name : rejectedNames) {
    if (core::isTensorIndexName(name)) {
      std::cerr << "FAIL: expected rejected tensor index name '" << name
                << "'\n";
      return false;
    }
  }

  return true;
}

static bool testIRTensorTypeMappingForExternCall() {
  static const char *kSource = R"(
    extern cov_tensor3 foo_cov(cov_tensor3)
    extern con_tensor3 foo_con(con_tensor3)
    field cov_tensor3 C[i,j,k]
    field con_tensor3 U[i,j,k]

    evolution E {
      dt C[i,j,k] = foo_cov(C[i,j,k])
      dt U[i,j,k] = foo_con(U[i,j,k])
    }
  )";

  Lexer lex(kSource);
  Parser parser(lex);
  Program prog = parser.parseProgram();
  SemanticAnalyzer sem(prog, CompilationMode::Symbolic);
  backend::ModuleIR mod = backend::BackendBuilder::build(prog, sem);

  if (mod.evolutions.empty() || mod.evolutions[0].equations.size() != 2) {
    std::cerr << "FAIL: expected two equations in IR extern mapping test\n";
    return false;
  }

  auto checkCall = [](const backend::EquationIR &eq, int retUp, int retDown,
                      int argUp, int argDown) {
    auto *call = dynamic_cast<const backend::CallIR *>(eq.rhs.get());
    if (!call)
      return false;
    if (call->returnType.up != retUp || call->returnType.down != retDown)
      return false;
    if (call->paramTypes.size() != 1)
      return false;
    if (call->paramTypes[0].up != argUp || call->paramTypes[0].down != argDown)
      return false;
    return true;
  };

  const auto &covEq = mod.evolutions[0].equations[0];
  const auto &conEq = mod.evolutions[0].equations[1];
  if (!checkCall(covEq, 0, 3, 0, 3)) {
    std::cerr << "FAIL: covariant extern tensor signature mapping mismatch\n";
    return false;
  }
  if (!checkCall(conEq, 3, 0, 3, 0)) {
    std::cerr << "FAIL: contravariant extern tensor signature mapping mismatch\n";
    return false;
  }

  return true;
}

int main() {
  bool ok = true;
  ok &= testConTensor3Lowering();
  ok &= testIndexSetPolicy();
  ok &= testIRTensorTypeMappingForExternCall();

  if (!ok)
    return 1;

  std::cout << "All unit tests passed\n";
  return 0;
}

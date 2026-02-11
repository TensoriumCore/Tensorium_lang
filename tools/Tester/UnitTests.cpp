#include "tensorium/Backend/BackendBuilder.hpp"
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
    if (field.up != 3 || field.down != 0) {
      std::cerr << "FAIL: expected field A variance up=3 down=0\n";
      return false;
    }
    return true;
  }

  std::cerr << "FAIL: field A not found in lowered backend module\n";
  return false;
}

int main() {
  bool ok = true;
  ok &= testConTensor3Lowering();

  if (!ok)
    return 1;

  std::cout << "All unit tests passed\n";
  return 0;
}

#include "tensorium/Lex/Lexer.hpp"
#include "tensorium/Parse/Parser.hpp"
#include "tensorium/Sema/Sema.hpp"
#include "tensorium/AST/ASTPrinter.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace tensorium;

static std::string readFile(const std::string &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("cannot open file: " + path);

  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

int main(int argc, char **argv) {
  bool dumpAST = false;

  if (argc < 2) {
    std::cerr << "usage: Tensorium_cc [--dump-ast] file1.tn [file2.tn ...]\n";
    return 1;
  }

  std::vector<std::string> files;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--dump-ast") {
      dumpAST = true;
    } else {
      files.push_back(arg);
    }
  }

  if (files.empty()) {
    std::cerr << "error: no input files\n";
    return 1;
  }

  try {
    for (const auto &path : files) {
      std::cout << "[Tensorium] parsing " << path << "\n";

      std::string src = readFile(path);

      Lexer lex(src.c_str());
      Parser parser(lex);
      Program prog = parser.parseProgram();

      SemanticAnalyzer sem(prog);

      if (dumpAST) {
        std::cout << "\n=== AST DUMP (" << path << ") ===\n";
        printProgram(prog);
        std::cout << "==============================\n";
      }

      std::cout << "[Tensorium] OK: " << path << "\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "Tensorium error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}

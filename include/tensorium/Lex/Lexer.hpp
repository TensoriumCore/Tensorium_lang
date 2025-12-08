#pragma once
#include "tensorium/Basic/Token.hpp"

namespace tensorium {
class Lexer {
  const char *src;
  int line = 1;
  int col = 1;
  void advanceChar();
public:
  explicit Lexer(const char *input);
  Token next();
};
}

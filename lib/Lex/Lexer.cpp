#include "tensorium/Lex/Lexer.hpp"
#include <cctype>

namespace tensorium {
Lexer::Lexer(const char *input) : src(input) {}

void Lexer::advanceChar() {
  if (*src == '\n') {
    ++line;
    col = 1;
  } else {
    ++col;
  }
  ++src;
}

Token Lexer::next() {
  while (*src) {
    if (std::isspace((unsigned char)*src)) {
      advanceChar();
      continue;
    }
    if (*src == '#') {
      while (*src && *src != '\n')
        ++src;
      continue;
    }
    break;
  }
  if (!*src)
    return {TokenType::End, "", line, col};

  char c = *src;
  switch (c) {
  case '(':
    advanceChar();
    return {TokenType::LParen, "(", line, col - 1};
  case ')':
    advanceChar();
    return {TokenType::RParen, ")", line, col - 1};
  case '{':
    advanceChar();
    return {TokenType::LBrace, "{", line, col - 1};
  case '}':
    advanceChar();
    return {TokenType::RBrace, "}", line, col - 1};
  case '[':
    advanceChar();
    return {TokenType::LBracket, "[", line, col - 1};
  case ']':
    advanceChar();
    return {TokenType::RBracket, "]", line, col - 1};
  case ',':
    advanceChar();
    return {TokenType::Comma, ",", line, col - 1};
  case '=':
    advanceChar();
    return {TokenType::Equals, "=", line, col - 1};
  case '+':
    advanceChar();
    return {TokenType::Plus, "+", line, col - 1};
  case '-':
    advanceChar();
    return {TokenType::Minus, "-", line, col - 1};
  case '*':
    advanceChar();
    return {TokenType::Star, "*", line, col - 1};
  case '/':
    advanceChar();
    return {TokenType::Slash, "/", line, col - 1};
  case '^':
    advanceChar();
    return {TokenType::Caret, "^", line, col - 1};
  }

  if (isdigit((unsigned char)c) ||
      (c == '.' && isdigit((unsigned char)*(src + 1)))) {
    const char *s = src;
    while (isdigit((unsigned char)*src) || *src == '.')
      ++src;
    return {TokenType::Number, std::string(s, src), line, col};
  }

  if (isalpha((unsigned char)c)) {
    const char *s = src;
    while (isalnum((unsigned char)*src) || *src == '_')
      ++src;
    std::string t(s, src);
    if (t == "spacetime")
      return {TokenType::KwSpacetime, t, line, col};
    if (t == "metric")
      return {TokenType::KwMetric, t, line, col};
    if (t == "evolution")
      return {TokenType::KwEvolution, t, line, col};
    if (t == "dt")
      return {TokenType::KwDt, t, line, col};
    if (t == "field")
      return {TokenType::KwField, t, line, col};
    if (t == "scalar")
      return {TokenType::KwScalar, t, line, col};
    if (t == "vector")
      return {TokenType::KwVector, t, line, col};
    if (t == "covector")
      return {TokenType::KwCovector, t, line, col};
    if (t == "cov_tensor2")
      return {TokenType::KwCovTensor2, t, line, col};
    if (t == "con_tensor2")
      return {TokenType::KwConTensor2, t, line, col};
    if (t == "cov_tensor3")
      return {TokenType::KwCovTensor3, t, line, col};
    if (t == "con_tensor3")
      return {TokenType::KwConTensor3, t, line, col};
    if (t == "cov_tensor4")
      return {TokenType::KwCovTensor4, t, line, col};
    if (t == "con_tensor4")
      return {TokenType::KwConTensor4, t, line, col};
    if (t == "simulation")
      return {TokenType::KwSimulation, t, line, col};
    if (t == "time")
      return {TokenType::KwTime, t, line, col};
    if (t == "spatial")
      return {TokenType::KwSpatial, t, line, col};
    return {TokenType::Identifier, t, line, col};
  }
  std::string u(1, c);
  advanceChar();
  return {TokenType::Unknown, u, line, col - 1};
}
} // namespace tensorium

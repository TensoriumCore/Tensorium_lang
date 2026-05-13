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
    if (*src == '#' || (*src == '/' && *(src + 1) == '/')) {
      while (*src && *src != '\n')
        advanceChar();
      continue;
    }
    break;
  }
  if (!*src)
    return {TokenType::End, "", line, col};

  char c = *src;
  const int tokLine = line;
  const int tokCol = col;
  switch (c) {
  case '(':
    advanceChar();
    return {TokenType::LParen, "(", tokLine, tokCol};
  case ')':
    advanceChar();
    return {TokenType::RParen, ")", tokLine, tokCol};
  case '{':
    advanceChar();
    return {TokenType::LBrace, "{", tokLine, tokCol};
  case '}':
    advanceChar();
    return {TokenType::RBrace, "}", tokLine, tokCol};
  case '[':
    advanceChar();
    return {TokenType::LBracket, "[", tokLine, tokCol};
  case ']':
    advanceChar();
    return {TokenType::RBracket, "]", tokLine, tokCol};
  case ',':
    advanceChar();
    return {TokenType::Comma, ",", tokLine, tokCol};
  case ';':
    advanceChar();
    return {TokenType::Semicolon, ";", tokLine, tokCol};
  case '=':
    advanceChar();
    return {TokenType::Equals, "=", tokLine, tokCol};
  case '+':
    advanceChar();
    return {TokenType::Plus, "+", tokLine, tokCol};
  case '-':
    if (*(src + 1) == '>') {
      advanceChar();
      advanceChar();
      return {TokenType::Arrow, "->", tokLine, tokCol};
    }
    advanceChar();
    return {TokenType::Minus, "-", tokLine, tokCol};
  case '*':
    advanceChar();
    return {TokenType::Star, "*", tokLine, tokCol};
  case '/':
    advanceChar();
    return {TokenType::Slash, "/", tokLine, tokCol};
  case '^':
    advanceChar();
    return {TokenType::Caret, "^", tokLine, tokCol};
  }

  if (isdigit((unsigned char)c) ||
      (c == '.' && isdigit((unsigned char)*(src + 1)))) {
    const char *start = src;
    while (isdigit((unsigned char)*src) || *src == '.')
      advanceChar();
    return {TokenType::Number, std::string(start, src), tokLine, tokCol};
  }

  if (isalpha((unsigned char)c)) {
    const char *start = src;
    while (isalnum((unsigned char)*src) || *src == '_')
      advanceChar();
    std::string t(start, src);
    if (t == "spacetime")
      return {TokenType::KwSpacetime, t, tokLine, tokCol};
    if (t == "metric")
      return {TokenType::KwMetric, t, tokLine, tokCol};
    if (t == "params")
      return {TokenType::KwParams, t, tokLine, tokCol};
    if (t == "inverse_metric")
      return {TokenType::KwInverseMetric, t, tokLine, tokCol};
    if (t == "evolution")
      return {TokenType::KwEvolution, t, tokLine, tokCol};
    if (t == "dt")
      return {TokenType::KwDt, t, tokLine, tokCol};
    if (t == "print")
      return {TokenType::KwPrint, t, tokLine, tokCol};
    if (t == "field")
      return {TokenType::KwField, t, tokLine, tokCol};
    if (t == "extern")
      return {TokenType::KwExtern, t, tokLine, tokCol};
    if (t == "scalar")
      return {TokenType::KwScalar, t, tokLine, tokCol};
    if (t == "vector")
      return {TokenType::KwVector, t, tokLine, tokCol};
    if (t == "covector")
      return {TokenType::KwCovector, t, tokLine, tokCol};
    if (t == "cov_tensor2")
      return {TokenType::KwCovTensor2, t, tokLine, tokCol};
    if (t == "con_tensor2")
      return {TokenType::KwConTensor2, t, tokLine, tokCol};
    if (t == "cov_tensor3")
      return {TokenType::KwCovTensor3, t, tokLine, tokCol};
    if (t == "con_tensor3")
      return {TokenType::KwConTensor3, t, tokLine, tokCol};
    if (t == "cov_tensor4")
      return {TokenType::KwCovTensor4, t, tokLine, tokCol};
    if (t == "con_tensor4")
      return {TokenType::KwConTensor4, t, tokLine, tokCol};
    if (t == "simulation")
      return {TokenType::KwSimulation, t, tokLine, tokCol};
    if (t == "initial_data")
      return {TokenType::KwInitialData, t, tokLine, tokCol};
    if (t == "metric4")
      return {TokenType::KwMetric4, t, tokLine, tokCol};
    if (t == "time")
      return {TokenType::KwTime, t, tokLine, tokCol};
    if (t == "spatial")
      return {TokenType::KwSpatial, t, tokLine, tokCol};
    return {TokenType::Identifier, t, tokLine, tokCol};
  }
  std::string u(1, c);
  advanceChar();
  return {TokenType::Unknown, u, tokLine, tokCol};
}
} // namespace tensorium

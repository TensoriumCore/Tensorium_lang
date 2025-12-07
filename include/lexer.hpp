#pragma once
#include "tokens.hpp"
#include <cctype>

class Lexer {
  const char *src;
  int line = 1;
  int col = 1;

  void advanceChar() {
    if (*src == '\n') {
      ++line;
      col = 1;
    } else {
      ++col;
    }
    ++src;
  }

public:
  explicit Lexer(const char *input) : src(input) {}

  Token next() {
    while (*src) {
      if (std::isspace(static_cast<unsigned char>(*src))) {
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
    default:
      break;
    }

    if (std::isdigit(static_cast<unsigned char>(c)) ||
        (c == '.' && std::isdigit(static_cast<unsigned char>(*(src + 1))))) {

      const char *start = src;
      while (std::isdigit(static_cast<unsigned char>(*src)) || *src == '.')
        ++src;

      return {TokenType::Number, std::string(start, src), line, col};
    }

    if (std::isalpha(static_cast<unsigned char>(c))) {
      const char *start = src;
      while (std::isalnum(static_cast<unsigned char>(*src)) || *src == '_')
        ++src;

      std::string text(start, src);

      if (text == "spacetime")
        return {TokenType::KwSpacetime, text, line, col};
      if (text == "metric")
        return {TokenType::KwMetric, text, line, col};
      if (text == "decompose")
        return {TokenType::KwDecompose, text, line, col};
      if (text == "coords")
        return {TokenType::KwCoords, text, line, col};
      if (text == "params")
        return {TokenType::KwParams, text, line, col};
      if (text == "signature")
        return {TokenType::KwSignature, text, line, col};
      if (text == "lapse")
        return {TokenType::KwLapse, text, line, col};
      if (text == "shift")
        return {TokenType::KwShift, text, line, col};
      if (text == "spatial")
        return {TokenType::KwSpatial, text, line, col};
      if (text == "extrinsic")
        return {TokenType::KwExtrinsic, text, line, col};
      if (text == "field")
        return {TokenType::KwField, text, line, col};
      if (text == "scalar")
        return {TokenType::KwScalar, text, line, col};
      if (text == "vector")
        return {TokenType::KwVector, text, line, col};
      if (text == "tensor2")
        return {TokenType::KwTensor2, text, line, col};
      if (text == "evolution")
        return {TokenType::KwEvolution, text, line, col};
      if (text == "dt")
        return {TokenType::KwDt, text, line, col};
      return {TokenType::Identifier, text, line, col};
    }

    std::string txt(1, c);
    advanceChar();
    return {TokenType::Unknown, txt, line, col - 1};
  }
};

#pragma once
#include <string>

namespace tensorium {
enum class TokenType {
  End,
  Identifier,
  Number,
  LParen,
  RParen,
  LBrace,
  RBrace,
  LBracket,
  RBracket,
  Comma,
  Equals,
  Plus,
  Minus,
  Star,
  Slash,
  Caret,
  KwSpacetime,
  KwMetric,
  KwDecompose,
  KwCoords,
  KwParams,
  KwSignature,
  KwLapse,
  KwShift,
  KwSpatial,
  KwExtrinsic,
  KwField,
  KwScalar,
  KwVector,
  KwTensor2,
  KwCovector,
  KwCovTensor2,
  KwConTensor2,
  KwEq,
  KwEvolution,
  KwDt,
  Unknown
};

struct Token {
  TokenType type;
  std::string text;
  int line;
  int column;
};
} // namespace tensorium

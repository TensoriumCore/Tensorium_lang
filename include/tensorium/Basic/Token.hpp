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
  KwExtern,
  KwScalar,
  KwVector,
  KwTensor2,
  KwCovector,
  KwCovTensor2,
  KwConTensor2,
  KwCovTensor3,
  KwConTensor3,
  KwCovTensor4,
  KwConTensor4,
  KwEq,
  KwEvolution,
  KwDt,
  KwSimulation,
  KwTime,
  Unknown
};

struct Token {
  TokenType type;
  std::string text;
  int line;
  int column;
};
} // namespace tensorium

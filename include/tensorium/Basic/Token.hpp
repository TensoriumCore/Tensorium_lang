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
  Semicolon,
  Arrow,
  Equals,
  Plus,
  Minus,
  Star,
  Slash,
  Caret,
  KwSpacetime,
  KwMetric,
  KwParams,
  KwSpatial,
  KwField,
  KwExtern,
  KwScalar,
  KwVector,
  KwCovector,
  KwCovTensor2,
  KwConTensor2,
  KwCovTensor3,
  KwConTensor3,
  KwCovTensor4,
  KwConTensor4,
  KwInverseMetric,
  KwEvolution,
  KwDt,
  KwSimulation,
  KwInitialData,
  KwMetric4,
  KwTime,
  Unknown
};

struct Token {
  TokenType type;
  std::string text;
  int line;
  int column;
};

inline const char *tokenTypeName(TokenType type) {
  switch (type) {
  case TokenType::End:
    return "end of file";
  case TokenType::Identifier:
    return "identifier";
  case TokenType::Number:
    return "number";
  case TokenType::LParen:
    return "'('";
  case TokenType::RParen:
    return "')'";
  case TokenType::LBrace:
    return "'{'";
  case TokenType::RBrace:
    return "'}'";
  case TokenType::LBracket:
    return "'['";
  case TokenType::RBracket:
    return "']'";
  case TokenType::Comma:
    return "','";
  case TokenType::Semicolon:
    return "';'";
  case TokenType::Arrow:
    return "'->'";
  case TokenType::Equals:
    return "'='";
  case TokenType::Plus:
    return "'+'";
  case TokenType::Minus:
    return "'-'";
  case TokenType::Star:
    return "'*'";
  case TokenType::Slash:
    return "'/'";
  case TokenType::Caret:
    return "'^'";
  case TokenType::KwSpacetime:
    return "'spacetime'";
  case TokenType::KwMetric:
    return "'metric'";
  case TokenType::KwParams:
    return "'params'";
  case TokenType::KwSpatial:
    return "'spatial'";
  case TokenType::KwField:
    return "'field'";
  case TokenType::KwExtern:
    return "'extern'";
  case TokenType::KwScalar:
    return "'scalar'";
  case TokenType::KwVector:
    return "'vector'";
  case TokenType::KwCovector:
    return "'covector'";
  case TokenType::KwCovTensor2:
    return "'cov_tensor2'";
  case TokenType::KwConTensor2:
    return "'con_tensor2'";
  case TokenType::KwCovTensor3:
    return "'cov_tensor3'";
  case TokenType::KwConTensor3:
    return "'con_tensor3'";
  case TokenType::KwCovTensor4:
    return "'cov_tensor4'";
  case TokenType::KwConTensor4:
    return "'con_tensor4'";
  case TokenType::KwInverseMetric:
    return "'inverse_metric'";
  case TokenType::KwEvolution:
    return "'evolution'";
  case TokenType::KwDt:
    return "'dt'";
  case TokenType::KwSimulation:
    return "'simulation'";
  case TokenType::KwInitialData:
    return "'initial_data'";
  case TokenType::KwMetric4:
    return "'metric4'";
  case TokenType::KwTime:
    return "'time'";
  case TokenType::Unknown:
    return "unknown token";
  }
  return "unknown token";
}
} // namespace tensorium

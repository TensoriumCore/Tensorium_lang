#include "tensorium/Parse/Parser.hpp"
#include "tensorium/Basic/Diagnostics.hpp"
#include "tensorium/Basic/Token.hpp"

#include <algorithm>
#include <string>

namespace tensorium {
Parser::Parser(Lexer &l) : lex(l) { advance(); }
void Parser::advance() { cur = lex.next(); }
void Parser::expect(TokenType type) {
  if (cur.type != type) {
    std::string got;
    if (cur.type == TokenType::End) {
      got = "end of file";
    } else if (cur.type == TokenType::Unknown) {
      got = "unknown token '" + cur.text + "'";
    } else {
      got = "'" + cur.text + "'";
    }
    syntaxError("expected " + std::string(tokenTypeName(type)) + ", got " + got);
  }
  advance();
}
void Parser::syntaxError(const std::string &msg) {
  SourceLocation loc;
  loc.line = std::max(1, cur.line);
  loc.column = std::max(1, cur.column);
  loc.length = std::max<int>(1, static_cast<int>(cur.text.size()));
  throw DiagnosticError(DiagnosticLevel::Error, msg, loc, "E0001");
}
} // namespace tensorium

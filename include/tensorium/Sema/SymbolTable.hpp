#pragma once
#include "tensorium/Basic/TensorSignature.hpp"
#include <optional>
#include <string>
#include <unordered_map>

namespace tensorium {

enum class SymbolKind { Field, Parameter, Local, Coordinate };

struct Symbol {
  std::string name;
  SymbolKind kind;
  TensorSignature signature;
  int coordIndex = -1;
};

class SymbolTable {
  std::unordered_map<std::string, Symbol> symbols;

public:
  void insert(const Symbol &sym) { symbols[sym.name] = sym; }

  std::optional<Symbol> lookup(const std::string &name) const {
    auto it = symbols.find(name);
    if (it != symbols.end())
      return it->second;
    return std::nullopt;
  }
};

} // namespace tensorium

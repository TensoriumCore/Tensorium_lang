#pragma once
#include "../ast.hpp"
#include "indexed_ast.hpp"
#include <stdexcept>
#include <unordered_map>

class SemanticAnalyzer {
  std::unordered_map<std::string, int> coordIndex;
  std::unordered_map<std::string, bool> locals;
  std::unordered_map<std::string, TensorKind> fields;

  int resolveIndex(const std::string &name) {
    auto it = coordIndex.find(name);
    if (it == coordIndex.end())
      throw std::runtime_error("Unknown tensor index: " + name);
    return it->second;
  }

  std::unique_ptr<IndexedExpr> transformExpr(const Expr *e) {
    if (auto n = dynamic_cast<const NumberExpr *>(e))
      return std::make_unique<IndexedNumber>(n->value);

    if (auto v = dynamic_cast<const VarExpr *>(e)) {
      if (auto it = coordIndex.find(v->name); it != coordIndex.end())
        return std::make_unique<IndexedVar>(v->name, IndexedVarKind::Coordinate,
                                            it->second);

      if (locals.count(v->name))
        return std::make_unique<IndexedVar>(v->name, IndexedVarKind::Local);

      if (fields.count(v->name))
        return std::make_unique<IndexedVar>(v->name, IndexedVarKind::Field);

      return std::make_unique<IndexedVar>(v->name, IndexedVarKind::Parameter);
    }

    if (auto b = dynamic_cast<const BinaryExpr *>(e))
      return std::make_unique<IndexedBinary>(b->op, transformExpr(b->lhs.get()),
                                             transformExpr(b->rhs.get()));

    if (auto p = dynamic_cast<const ParenExpr *>(e))
      return transformExpr(p->inner.get());

    if (auto iv = dynamic_cast<const IndexedVarExpr *>(e)) {
      auto it = fields.find(iv->base);
      if (it == fields.end())
        throw std::runtime_error("Unknown indexed tensor: " + iv->base);

      TensorKind kind = it->second;

      size_t expected = (kind == TensorKind::Scalar   ? 0
                         : kind == TensorKind::Vector ? 1
                                                      : 2);

      if (iv->indices.size() != expected)
        throw std::runtime_error("Tensor '" + iv->base + "' expects " +
                                 std::to_string(expected) + " indices, got " +
                                 std::to_string(iv->indices.size()));

      auto out = std::make_unique<IndexedVar>(iv->base, IndexedVarKind::Field);

      for (auto &idx : iv->indices) {
        int off = resolveIndex(idx); // -1 pour indices abstraits i,j,...
        out->tensorIndices.push_back(off);
        out->tensorIndexNames.push_back(idx); // <--- garder le nom "i","j"
      }

      return out;
    }
    if (auto c = dynamic_cast<const CallExpr *>(e)) {
      auto out = std::make_unique<IndexedCall>();
      out->callee = c->callee;
      for (auto &arg : c->args)
        out->args.push_back(transformExpr(arg.get()));
      return out;
    }

    throw std::runtime_error("Unsupported expr type in semantic analysis");
  }

public:
  explicit SemanticAnalyzer(const Program &prog) {
    // Fields environment
    for (const auto &f : prog.fields) {
      if (fields.count(f.name))
        throw std::runtime_error("Field redeclared: " + f.name);
      fields[f.name] = f.kind;
    }

    // Locals from all metrics: entries without indices
    for (const auto &m : prog.metrics) {
      for (const auto &entry : m.entries) {
        if (entry.lhs.indices.empty())
          locals[entry.lhs.base] = true;
      }
    }
  }

  IndexedMetric analyzeMetric(const MetricDecl &decl) {
    coordIndex.clear();

    IndexedMetric out;
    out.name = decl.name;
    out.rank = 2;
    out.coords = decl.indices;

    for (size_t i = 0; i < decl.indices.size(); ++i)
      coordIndex[decl.indices[i]] = static_cast<int>(i);

    for (const auto &entry : decl.entries) {
      IndexedAssignment a;
      a.tensor = entry.lhs.base;

      if (!entry.lhs.indices.empty()) {
        if (entry.lhs.indices.size() != 2)
          throw std::runtime_error("Metric tensor '" + entry.lhs.base +
                                   "' must have 2 indices (got " +
                                   std::to_string(entry.lhs.indices.size()) +
                                   ")");

        for (const auto &idx : entry.lhs.indices)
          a.indexOffsets.push_back(resolveIndex(idx));
      }

      a.rhs = transformExpr(entry.rhs.get());
      out.assignments.push_back(std::move(a));
    }

    return out;
  }

  IndexedEvolution analyzeEvolution(const EvolutionDecl &evo) {
    coordIndex.clear();

    IndexedEvolution out;
    out.name = evo.name;

    for (const auto &eq : evo.equations) {
      auto it = fields.find(eq.fieldName);
      if (it == fields.end())
        throw std::runtime_error("Unknown field in evolution: " + eq.fieldName);

      TensorKind kind = it->second;
      size_t expectedRank = 0;
      switch (kind) {
      case TensorKind::Scalar:
        expectedRank = 0;
        break;
      case TensorKind::Vector:
        expectedRank = 1;
        break;
      case TensorKind::Tensor2:
        expectedRank = 2;
        break;
      }

      if (eq.indices.size() != expectedRank) {
        throw std::runtime_error(
            "Wrong number of indices in evolution for field '" + eq.fieldName +
            "': expected " + std::to_string(expectedRank) + ", got " +
            std::to_string(eq.indices.size()));
      }
      for (const auto &idx : eq.indices)
        coordIndex[idx] = -1;
      IndexedEvolutionEq ie;
      ie.fieldName = eq.fieldName;
      ie.indices = eq.indices;

      ie.rhs = transformExpr(eq.rhs.get());
      out.equations.push_back(std::move(ie));
    }

    return out;
  }
};

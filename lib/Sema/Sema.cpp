#include "tensorium/Sema/Sema.hpp"
#include "tensorium/Sema/tensor_type_checker.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace tensorium {

void SemanticAnalyzer::validateSpatialIndex(const std::string &idx) {
  if (!SPATIAL_INDICES.count(idx)) {
    throw std::runtime_error("Invalid tensor index '" + idx +
                             "'. Allowed: {i, j, k, l, m, n}.");
  }
}

int SemanticAnalyzer::resolveIndex(const std::string &name) {
  auto it = coordIndex.find(name);
  if (it == coordIndex.end())
    throw std::runtime_error("Unknown tensor index: " + name);
  return it->second;
}

std::unique_ptr<IndexedExpr> SemanticAnalyzer::transformExpr(const Expr *e) {
  if (auto n = dynamic_cast<const NumberExpr *>(e))
    return std::make_unique<IndexedNumber>(n->value);

  if (auto v = dynamic_cast<const VarExpr *>(e)) {
    if (auto it = coordIndex.find(v->name); it != coordIndex.end()) {
      auto iv =
          std::make_unique<IndexedVar>(v->name, IndexedVarKind::Coordinate);
      iv->coordIndex = it->second;
      iv->tensorKind = TensorKind::Scalar;
      return iv;
    }

    if (locals.count(v->name)) {
      auto iv = std::make_unique<IndexedVar>(v->name, IndexedVarKind::Local);
      iv->tensorKind = TensorKind::Scalar;
      return iv;
    }

    if (auto itf = fields.find(v->name); itf != fields.end()) {
      const FieldDecl *fd = itf->second;
      auto iv = std::make_unique<IndexedVar>(v->name, IndexedVarKind::Field);
      iv->tensorKind = fd->kind;
      iv->up = fd->up;
      iv->down = fd->down;
      return iv;
    }

    auto iv = std::make_unique<IndexedVar>(v->name, IndexedVarKind::Parameter);
    iv->tensorKind = TensorKind::Scalar;
    return iv;
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

    const FieldDecl *fd = it->second;
    size_t expected = static_cast<size_t>(fd->up + fd->down);

    if (iv->indices.size() != expected)
      throw std::runtime_error("Tensor '" + iv->base + "' expects " +
                               std::to_string(expected) + " indices, got " +
                               std::to_string(iv->indices.size()));

    auto out = std::make_unique<IndexedVar>(iv->base, IndexedVarKind::Field);
    out->tensorKind = fd->kind;
    out->up = fd->up;
    out->down = fd->down;

    for (auto &idx : iv->indices) {
      if (!coordIndex.count(idx)) {
        validateSpatialIndex(idx);
        coordIndex[idx] = -2;
      }
      int off = resolveIndex(idx);
      out->tensorIndices.push_back(off);
      out->tensorIndexNames.push_back(idx);
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

  throw std::runtime_error("Unsupported expr in semantic analyzer");
}

SemanticAnalyzer::SemanticAnalyzer(const Program &p) : prog(p) {
  for (const auto &f : prog.fields) {
    if (fields.count(f.name))
      throw std::runtime_error("Field redeclared: " + f.name);
    fields[f.name] = &f;
  }

  for (const auto &m : prog.metrics) {
    for (const auto &entry : m.entries) {
      if (entry.lhs.indices.empty())
        locals[entry.lhs.base] = true;
    }
  }
  for (const auto &m : prog.metrics) {
    FieldDecl fd;
    fd.kind = TensorKind::CovTensor2;
    fd.name = m.name;
    fd.up = 0;
    fd.down = 2;
    syntheticMetricFields.push_back(fd);
    fields[m.name] = &syntheticMetricFields.back();
  }
  if (prog.simulation) {
    validateSimulation(*prog.simulation);
  }
}

IndexedMetric SemanticAnalyzer::analyzeMetric(const MetricDecl &decl) {
  coordIndex.clear();

  IndexedMetric out;
  out.name = decl.name;
  out.rank = 2;
  out.coords = decl.indices;

  for (size_t i = 0; i < decl.indices.size(); ++i)
    coordIndex[decl.indices[i]] = static_cast<int>(i);

  TensorTypeChecker checker;

  for (const auto &entry : decl.entries) {
    IndexedAssignment a;
    a.tensor = entry.lhs.base;

    if (!entry.lhs.indices.empty()) {
      if (entry.lhs.indices.size() != 2)
        throw std::runtime_error(
            "Metric tensor '" + entry.lhs.base + "' must have 2 indices (got " +
            std::to_string(entry.lhs.indices.size()) + ")");
      for (const auto &idx : entry.lhs.indices)
        a.indexOffsets.push_back(resolveIndex(idx));
    }

    a.rhs = transformExpr(entry.rhs.get());
    checker.checkMetricAssignment(a);
    out.assignments.push_back(std::move(a));
  }

  return out;
}

struct IndexCollector {
  std::unordered_map<std::string, int> &counter;
  IndexCollector(std::unordered_map<std::string, int> &c) : counter(c) {}

  void walk(const IndexedExpr *expr) {
    if (auto v = dynamic_cast<const IndexedVar *>(expr)) {
      for (auto &idx : v->tensorIndexNames)
        counter[idx]++;
    }
    if (auto b = dynamic_cast<const IndexedBinary *>(expr)) {
      walk(b->lhs.get());
      walk(b->rhs.get());
    }
    if (auto c = dynamic_cast<const IndexedCall *>(expr)) {
      for (auto &arg : c->args)
        walk(arg.get());
    }
  }
};

IndexedEvolution SemanticAnalyzer::analyzeEvolution(const EvolutionDecl &evo) {
  coordIndex.clear();

  IndexedEvolution out;
  out.name = evo.name;

  for (const auto &eq : evo.equations)
    for (const auto &idx : eq.indices) {
      validateSpatialIndex(idx);
      coordIndex[idx] = -1;
    }

  for (const auto &tmp : evo.tempAssignments) {
    if (!tmp.lhs.indices.empty()) {
      continue;
    }

    if (fields.count(tmp.lhs.base)) {
      throw std::runtime_error("Cannot redeclare field '" + tmp.lhs.base +
                               "' as local");
    }

    locals[tmp.lhs.base] = true;
  }

  TensorTypeChecker checker;

  for (const auto &eq : evo.equations) {

    auto it = fields.find(eq.fieldName);
    if (it == fields.end())
      throw std::runtime_error("Unknown field in evolution: " + eq.fieldName);

    const FieldDecl *fd = it->second;
    size_t expectedRank = static_cast<size_t>(fd->up + fd->down);

    if (eq.indices.size() != expectedRank) {
      throw std::runtime_error(
          "Wrong number of indices in evolution for field '" + eq.fieldName +
          "': expected " + std::to_string(expectedRank) + ", got " +
          std::to_string(eq.indices.size()));
    }

    indexUseCount.clear();
    lhsIndices.clear();

    for (auto &idx : eq.indices)
      lhsIndices.insert(idx);

    IndexedEvolutionEq ie;
    ie.fieldName = eq.fieldName;
    ie.indices = eq.indices;
    ie.rhs = transformExpr(eq.rhs.get());

    IndexCollector collector(indexUseCount);
    collector.walk(ie.rhs.get());

    for (auto &[idx, count] : indexUseCount) {
      validateSpatialIndex(idx);

      if (count == 1 && !lhsIndices.count(idx)) {
        throw std::runtime_error("Free index '" + idx +
                                 "' appears only in RHS and not LHS.");
      }

      if (count > 2) {
        throw std::runtime_error("Ambiguous contraction: index '" + idx +
                                 "' appears " + std::to_string(count) +
                                 " times.");
      }
    }


TensorType lhsType = {fd->up, fd->down};
checker.checkAssignmentVariance(lhsType, ie.indices, ie.rhs.get());


    out.equations.push_back(std::move(ie));
  }

  for (const auto &tmp : evo.tempAssignments) {
    IndexedAssignment ia;
    ia.tensor = tmp.lhs.base;
    for (auto &idx : tmp.lhs.indices)
      ia.indexOffsets.push_back(resolveIndex(idx));
    ia.rhs = transformExpr(tmp.rhs.get());
    out.temp.push_back(std::move(ia));
  }

  return out;
}

void SemanticAnalyzer::validateSimulation(const SimulationConfig &sim) {
  if (sim.dimension <= 0) {
    throw std::runtime_error("simulation dimension must be >= 1");
  }

  if ((int)sim.resolution.size() != sim.dimension) {
    throw std::runtime_error(
        "resolution size (" + std::to_string(sim.resolution.size()) +
        ") does not match dimension (" + std::to_string(sim.dimension) + ")");
  }

  for (int r : sim.resolution) {
    if (r <= 0)
      throw std::runtime_error("resolution entries must be > 0");
  }

  if (sim.time.dt <= 0.0) {
    throw std::runtime_error("time.dt must be > 0");
  }

  if (sim.spatial.scheme == SpatialScheme::FiniteDifference) {
    if (sim.spatial.order < 2)
      throw std::runtime_error("FD order must be >= 2");

    if (sim.spatial.order % 2 != 0)
      throw std::runtime_error("FD order must be even");
  }

  if (sim.spatial.scheme == SpatialScheme::Spectral) {
    if (sim.spatial.order != 0) {
      throw std::runtime_error(
          "spectral scheme does not use finite-difference order");
    }
  }
}
} // namespace tensorium

#include "tensorium/Sema/Sema.hpp"
#include <set>
#include <stdexcept>

namespace tensorium {

static bool exprStructuralEqual(const Expr *lhs, const Expr *rhs) {
  if (!lhs || !rhs)
    return lhs == rhs;

  if (auto ln = dynamic_cast<const NumberExpr *>(lhs)) {
    auto rn = dynamic_cast<const NumberExpr *>(rhs);
    return rn && ln->value == rn->value;
  }
  if (auto lv = dynamic_cast<const VarExpr *>(lhs)) {
    auto rv = dynamic_cast<const VarExpr *>(rhs);
    return rv && lv->name == rv->name;
  }
  if (auto lb = dynamic_cast<const BinaryExpr *>(lhs)) {
    auto rb = dynamic_cast<const BinaryExpr *>(rhs);
    return rb && lb->op == rb->op &&
           exprStructuralEqual(lb->lhs.get(), rb->lhs.get()) &&
           exprStructuralEqual(lb->rhs.get(), rb->rhs.get());
  }
  if (auto lc = dynamic_cast<const CallExpr *>(lhs)) {
    auto rc = dynamic_cast<const CallExpr *>(rhs);
    if (!rc || lc->callee != rc->callee || lc->args.size() != rc->args.size())
      return false;
    for (size_t i = 0; i < lc->args.size(); ++i) {
      if (!exprStructuralEqual(lc->args[i].get(), rc->args[i].get()))
        return false;
    }
    return true;
  }
  if (auto lp = dynamic_cast<const ParenExpr *>(lhs)) {
    auto rp = dynamic_cast<const ParenExpr *>(rhs);
    return rp && exprStructuralEqual(lp->inner.get(), rp->inner.get());
  }
  if (auto li = dynamic_cast<const IndexedVarExpr *>(lhs)) {
    auto ri = dynamic_cast<const IndexedVarExpr *>(rhs);
    return ri && li->base == ri->base && li->indices == ri->indices;
  }
  return false;
}

static bool isCoordinateAlias(const std::string &name) {
  static const std::set<std::string> kCoordinateAliases = {
      "t", "x", "y", "z", "r", "rho", "theta", "phi"};
  return kCoordinateAliases.count(name) != 0;
}

static bool isAllowedCoordinateName(const SimulationConfig &sim,
                                    const std::string &name) {
  if (sim.coordinates == CoordinateSystem::Spherical) {
    if (name == "r")
      return sim.dimension >= 1;
    if (name == "theta")
      return sim.dimension >= 2;
    if (name == "phi")
      return sim.dimension >= 3;
    return false;
  }

  if (sim.coordinates == CoordinateSystem::Cartesian) {
    if (name == "x")
      return sim.dimension >= 1;
    if (name == "y")
      return sim.dimension >= 2;
    if (name == "z")
      return sim.dimension >= 3;
    return false;
  }

  if (sim.coordinates == CoordinateSystem::Cylindrical) {
    if (name == "rho")
      return sim.dimension >= 1;
    if (name == "phi")
      return sim.dimension >= 2;
    if (name == "z")
      return sim.dimension >= 3;
    return false;
  }

  return false;
}

void SemanticAnalyzer::validateInitialDataExpr(const Expr *expr,
                                               const std::string &context) {
  if (!expr)
    throw std::runtime_error("initial_data " + context +
                             " contains empty expression");

  if (auto n = dynamic_cast<const NumberExpr *>(expr)) {
    (void)n;
    return;
  }

  if (auto v = dynamic_cast<const VarExpr *>(expr)) {
    if (!prog.simulation)
      return;
    if (isCoordinateAlias(v->name) &&
        !isAllowedCoordinateName(*prog.simulation, v->name)) {
      throw std::runtime_error("initial_data " + context +
                               " uses coordinate '" + v->name +
                               "' incompatible with simulation coordinates");
    }
    return;
  }

  if (auto b = dynamic_cast<const BinaryExpr *>(expr)) {
    validateInitialDataExpr(b->lhs.get(), context);
    validateInitialDataExpr(b->rhs.get(), context);
    return;
  }

  if (auto p = dynamic_cast<const ParenExpr *>(expr)) {
    validateInitialDataExpr(p->inner.get(), context);
    return;
  }

  if (auto c = dynamic_cast<const CallExpr *>(expr)) {
    static const std::set<std::string> kAllowedScalarFns = {"sin", "sqrt"};
    if (kAllowedScalarFns.count(c->callee) == 0) {
      throw std::runtime_error("initial_data " + context +
                               " uses unsupported scalar function '" +
                               c->callee + "'");
    }
    for (const auto &arg : c->args)
      validateInitialDataExpr(arg.get(), context);
    return;
  }

  if (auto idx = dynamic_cast<const IndexedVarExpr *>(expr)) {
    throw std::runtime_error("initial_data " + context +
                             " must be scalar expression, got indexed term '" +
                             idx->base + "'");
  }

  throw std::runtime_error("initial_data " + context +
                           " contains unsupported expression node");
}

void SemanticAnalyzer::validateInitialData(const InitialDataDecl &init) {
  if (!prog.simulation) {
    throw std::runtime_error("initial_data requires simulation block");
  }

  if (init.hasMetric4) {
    if (init.metric4.indices.size() != 2) {
      throw std::runtime_error("initial_data metric4 expects exactly 2 indices");
    }
    if (init.metric4.components.size() != 4) {
      throw std::runtime_error("initial_data metric4 expects a 4x4 matrix");
    }
    for (size_t i = 0; i < init.metric4.components.size(); ++i) {
      if (init.metric4.components[i].size() != 4) {
        throw std::runtime_error("initial_data metric4 row " +
                                 std::to_string(i) +
                                 " must have 4 entries");
      }
      for (size_t j = 0; j < init.metric4.components[i].size(); ++j) {
        validateInitialDataExpr(
            init.metric4.components[i][j].get(),
            "metric4[" + std::to_string(i) + "," + std::to_string(j) + "]");
      }
    }

    if (init.enforceSymmetry) {
      for (size_t i = 0; i < 4; ++i) {
        for (size_t j = i + 1; j < 4; ++j) {
          if (!exprStructuralEqual(init.metric4.components[i][j].get(),
                                   init.metric4.components[j][i].get())) {
            throw std::runtime_error(
                "initial_data metric4 symmetry violation at (" +
                std::to_string(i) + "," + std::to_string(j) + ")");
          }
        }
      }
    }
  }

  if (init.hasDecomposed) {
    if (!init.decomposed.alpha) {
      throw std::runtime_error("initial_data requires alpha expression");
    }
    validateInitialDataExpr(init.decomposed.alpha.get(), "alpha");

    if (init.decomposed.beta.size() != 3) {
      throw std::runtime_error("initial_data beta must have 3 entries");
    }
    for (size_t i = 0; i < init.decomposed.beta.size(); ++i) {
      validateInitialDataExpr(init.decomposed.beta[i].get(),
                              "beta[" + std::to_string(i) + "]");
    }

    if (init.decomposed.gamma.size() != 3) {
      throw std::runtime_error("initial_data gamma must be 3x3");
    }
    for (size_t i = 0; i < init.decomposed.gamma.size(); ++i) {
      if (init.decomposed.gamma[i].size() != 3) {
        throw std::runtime_error("initial_data gamma row " +
                                 std::to_string(i) +
                                 " must have 3 entries");
      }
      for (size_t j = 0; j < init.decomposed.gamma[i].size(); ++j) {
        validateInitialDataExpr(
            init.decomposed.gamma[i][j].get(),
            "gamma[" + std::to_string(i) + "," + std::to_string(j) + "]");
      }
    }

    if (!init.decomposed.gammaU.empty()) {
      if (init.decomposed.gammaU.size() != 3) {
        throw std::runtime_error("initial_data gammaU must be 3x3");
      }
      for (size_t i = 0; i < init.decomposed.gammaU.size(); ++i) {
        if (init.decomposed.gammaU[i].size() != 3) {
          throw std::runtime_error("initial_data gammaU row " +
                                   std::to_string(i) +
                                   " must have 3 entries");
        }
        for (size_t j = 0; j < init.decomposed.gammaU[i].size(); ++j) {
          validateInitialDataExpr(
              init.decomposed.gammaU[i][j].get(),
              "gammaU[" + std::to_string(i) + "," + std::to_string(j) + "]");
        }
      }
    }
  }

  if (init.hasMetric4) {
    auto it = fields.find(init.metric4.name);
    if (it != fields.end()) {
      const FieldDecl *fd = it->second;
      const bool validMetricType =
          (fd->kind == TensorKind::Metric) ||
          (fd->kind == TensorKind::CovTensor2 && fd->up == 0 && fd->down == 2);
      if (!validMetricType) {
        throw std::runtime_error("initial_data metric4 target '" +
                                 init.metric4.name +
                                 "' must be metric or covariant rank-2 field");
      }
    }
  }

  if (init.split3p1.enabled) {
    if (!init.hasMetric4 && !init.hasDecomposed) {
      throw std::runtime_error(
          "split_3p1 mapping requires metric4 or decomposed initial_data definition");
    }

    auto validateTarget = [this](const TensorAccess &target, int expUp,
                                 int expDown, const std::string &label,
                                 bool allowMetricLike = false,
                                 bool allowInverseMetricLike = false) {
      auto it = fields.find(target.base);
      if (it == fields.end()) {
        throw std::runtime_error("split_3p1 target field '" + target.base +
                                 "' for " + label + " is not declared");
      }
      const FieldDecl *fd = it->second;

      bool typeOk = (fd->up == expUp && fd->down == expDown);
      if (allowMetricLike && fd->kind == TensorKind::Metric && expUp == 0 &&
          expDown == 2)
        typeOk = true;
      if (allowInverseMetricLike && fd->kind == TensorKind::InverseMetric &&
          expUp == 2 && expDown == 0)
        typeOk = true;
      if (!typeOk) {
        throw std::runtime_error("split_3p1 target '" + target.base +
                                 "' for " + label + " has wrong tensor type");
      }

      const size_t expectedRank = static_cast<size_t>(expUp + expDown);
      if (target.indices.size() != expectedRank) {
        throw std::runtime_error("split_3p1 target '" + target.base +
                                 "' for " + label +
                                 " has wrong number of indices");
      }
    };

    if (init.split3p1.hasAlpha) {
      validateTarget(init.split3p1.alphaTarget, 0, 0, "alpha");
    }
    if (init.split3p1.hasBeta) {
      validateTarget(init.split3p1.betaTarget, 0, 1, "beta");
    }
    if (init.split3p1.hasGamma) {
      validateTarget(init.split3p1.gammaTarget, 0, 2, "gamma", true, false);
    }
    if (init.split3p1.hasGammaU) {
      validateTarget(init.split3p1.gammaUTarget, 2, 0, "gammaU", false, true);
    }
  }
}

} // namespace tensorium

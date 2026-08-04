#include "tensorium/Sema/Sema.hpp"
#include <cmath>
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
    if (!prog.simulation) {
      throw std::runtime_error("initial_data " + context +
                               " requires simulation metadata");
    }
    if (isCoordinateAlias(v->name)) {
      if (!isAllowedCoordinateName(*prog.simulation, v->name)) {
        throw std::runtime_error("initial_data " + context +
                                 " uses coordinate '" + v->name +
                                 "' incompatible with simulation coordinates");
      }
      return;
    }
    if (params.count(v->name))
      return;
    throw std::runtime_error("initial_data " + context +
                             " uses unknown identifier '" + v->name +
                             "' (declare it in params { ... })");
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
  if ((init.hasMetric4 || init.hasDecomposed) && !prog.simulation) {
    throw std::runtime_error("analytic initial_data requires simulation block");
  }

  if (init.hasMetric4) {
    if (init.metric4.indices.size() != 2) {
      throw std::runtime_error(
          "initial_data metric4 expects exactly 2 indices");
    }
    if (init.metric4.components.size() != 4) {
      throw std::runtime_error("initial_data metric4 expects a 4x4 matrix");
    }
    for (size_t i = 0; i < init.metric4.components.size(); ++i) {
      if (init.metric4.components[i].size() != 4) {
        throw std::runtime_error("initial_data metric4 row " +
                                 std::to_string(i) + " must have 4 entries");
      }
      for (size_t j = 0; j < init.metric4.components[i].size(); ++j) {
        validateInitialDataExpr(init.metric4.components[i][j].get(),
                                "metric4[" + std::to_string(i) + "," +
                                    std::to_string(j) + "]");
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
        throw std::runtime_error("initial_data gamma row " + std::to_string(i) +
                                 " must have 3 entries");
      }
      for (size_t j = 0; j < init.decomposed.gamma[i].size(); ++j) {
        validateInitialDataExpr(init.decomposed.gamma[i][j].get(),
                                "gamma[" + std::to_string(i) + "," +
                                    std::to_string(j) + "]");
      }
    }

    if (!init.decomposed.gammaU.empty()) {
      if (init.decomposed.gammaU.size() != 3) {
        throw std::runtime_error("initial_data gammaU must be 3x3");
      }
      for (size_t i = 0; i < init.decomposed.gammaU.size(); ++i) {
        if (init.decomposed.gammaU[i].size() != 3) {
          throw std::runtime_error("initial_data gammaU row " +
                                   std::to_string(i) + " must have 3 entries");
        }
        for (size_t j = 0; j < init.decomposed.gammaU[i].size(); ++j) {
          validateInitialDataExpr(init.decomposed.gammaU[i][j].get(),
                                  "gammaU[" + std::to_string(i) + "," +
                                      std::to_string(j) + "]");
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
      throw std::runtime_error("split_3p1 mapping requires metric4 or "
                               "decomposed initial_data definition");
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
        throw std::runtime_error("split_3p1 target '" + target.base + "' for " +
                                 label + " has wrong tensor type");
      }

      const size_t expectedRank = static_cast<size_t>(expUp + expDown);
      if (target.indices.size() != expectedRank) {
        throw std::runtime_error("split_3p1 target '" + target.base + "' for " +
                                 label + " has wrong number of indices");
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

  if (init.hasConstraintProblem)
    validateConstraintProblem(init.constraintProblem);
}

void SemanticAnalyzer::validateConstraintProblem(
    const ConstraintProblemDecl &problem) {
  if (problem.name.empty())
    throw std::runtime_error("constraint problem requires a name");
  if (problem.domains.empty())
    throw std::runtime_error("constraint problem requires at least one domain");
  if (problem.unknowns.empty())
    throw std::runtime_error(
        "constraint problem requires at least one unknown");
  if (problem.equations.empty())
    throw std::runtime_error(
        "constraint problem requires at least one equation");
  if (problem.boundaries.empty())
    throw std::runtime_error(
        "constraint problem requires at least one boundary");
  if (!problem.hasSolve)
    throw std::runtime_error("constraint problem requires a solve block");

  static const std::set<std::string> kCoordinates = {"cartesian", "spherical",
                                                     "cylindrical"};
  static const std::set<std::string> kTopologies = {
      "ball", "shell", "compactified", "bispherical", "rectilinear"};
  static const std::set<std::string> kBases = {"chebyshev", "legendre",
                                               "fourier", "chebyshev_fourier",
                                               "legendre_fourier"};

  std::set<std::string> domainNames;
  std::string coordinateSystem;
  size_t domainDimension = 0;
  for (const auto &domain : problem.domains) {
    if (!domainNames.insert(domain.name).second)
      throw std::runtime_error("constraint domain redeclared: " + domain.name);
    if (!kCoordinates.count(domain.coordinates))
      throw std::runtime_error("unsupported constraint coordinate system '" +
                               domain.coordinates + "'");
    if (coordinateSystem.empty())
      coordinateSystem = domain.coordinates;
    else if (coordinateSystem != domain.coordinates)
      throw std::runtime_error(
          "all constraint domains must use the same coordinate system");
    if (!kTopologies.count(domain.topology))
      throw std::runtime_error("unsupported constraint topology '" +
                               domain.topology + "'");
    if (!kBases.count(domain.basis))
      throw std::runtime_error("unsupported spectral basis '" + domain.basis +
                               "'");
    if (domain.resolution.empty() || domain.resolution.size() > 3)
      throw std::runtime_error(
          "constraint domain resolution must have between 1 and 3 entries");
    if (domainDimension == 0)
      domainDimension = domain.resolution.size();
    else if (domainDimension != domain.resolution.size())
      throw std::runtime_error(
          "all constraint domains must have the same dimension");
    for (int resolution : domain.resolution) {
      if (resolution <= 0)
        throw std::runtime_error(
            "constraint domain resolution entries must be > 0");
    }
    if (!domain.bounds.empty()) {
      if (domain.topology == "compactified") {
        if (domain.bounds.size() != 1 || !std::isfinite(domain.bounds[0]) ||
            domain.bounds[0] <= 0.0) {
          throw std::runtime_error(
              "compactified domain bounds must contain one finite positive "
              "inner radius");
        }
      } else {
        if (domain.bounds.size() != 2)
          throw std::runtime_error(
              "finite constraint domain bounds must contain exactly two "
              "entries");
        if (!std::isfinite(domain.bounds[0]) ||
            !std::isfinite(domain.bounds[1]) ||
            domain.bounds[0] >= domain.bounds[1]) {
          throw std::runtime_error(
              "finite constraint domain bounds must be finite and strictly "
              "increasing");
        }
        if (domain.coordinates == "spherical" && domain.bounds[0] <= 0.0) {
          throw std::runtime_error(
              "spherical shell bounds must have a strictly positive radius");
        }
      }
    }
  }

  std::set<std::pair<std::string, std::string>> interfacePairs;
  std::set<std::string> interfaceInputs;
  std::set<std::string> interfaceOutputs;
  for (const auto &interface : problem.interfaces) {
    if (!domainNames.count(interface.innerDomain))
      throw std::runtime_error("constraint interface references unknown domain '" +
                               interface.innerDomain + "'");
    if (!domainNames.count(interface.outerDomain))
      throw std::runtime_error("constraint interface references unknown domain '" +
                               interface.outerDomain + "'");
    if (interface.innerDomain == interface.outerDomain)
      throw std::runtime_error("constraint interface cannot connect domain '" +
                               interface.innerDomain + "' to itself");
    if (!interfacePairs
             .insert({interface.innerDomain, interface.outerDomain})
             .second)
      throw std::runtime_error("constraint interface redeclared: " +
                               interface.innerDomain + " -> " +
                               interface.outerDomain);
    if (!interfaceInputs.insert(interface.innerDomain).second)
      throw std::runtime_error("constraint domain '" + interface.innerDomain +
                               "' has multiple outer interfaces");
    if (!interfaceOutputs.insert(interface.outerDomain).second)
      throw std::runtime_error("constraint domain '" + interface.outerDomain +
                               "' has multiple inner interfaces");
  }

  std::set<std::string> unknownNames;
  for (const auto &unknown : problem.unknowns) {
    if (!unknownNames.insert(unknown.name).second)
      throw std::runtime_error("constraint unknown redeclared: " +
                               unknown.name);
    const size_t rank =
        static_cast<size_t>(unknown.type.up + unknown.type.down);
    if (unknown.symmetric &&
        !((unknown.type.up == 0 && unknown.type.down == 2) ||
          (unknown.type.up == 2 && unknown.type.down == 0))) {
      throw std::runtime_error("symmetric constraint unknown '" +
                               unknown.name +
                               "' must be covariant or contravariant rank two");
    }
    if (unknown.indices.size() != rank)
      throw std::runtime_error("constraint unknown '" + unknown.name +
                               "' declares wrong number of indices");
    for (const auto &index : unknown.indices)
      validateSpatialIndex(index);
  }

  if (problem.cttReconstruction.enabled) {
    auto requireScalarUnknown = [&](const std::string &name,
                                    const std::string &role) {
      auto it = std::find_if(problem.unknowns.begin(), problem.unknowns.end(),
                             [&](const ConstraintUnknownDecl &unknown) {
                               return unknown.name == name;
                             });
      if (it == problem.unknowns.end()) {
        throw std::runtime_error("reconstruct ctt " + role +
                                 " references unknown symbol '" + name +
                                 "'");
      }
      if (it->type.up != 0 || it->type.down != 0) {
        throw std::runtime_error("reconstruct ctt " + role + " '" + name +
                                 "' must be a scalar radial unknown");
      }
    };
    requireScalarUnknown(problem.cttReconstruction.conformalFactor,
                         "conformal_factor");
    requireScalarUnknown(problem.cttReconstruction.radialVectorPotential,
                         "radial_vector");
    if (!problem.cttReconstruction.meanCurvature)
      throw std::runtime_error(
          "reconstruct ctt requires a mean_curvature expression");
  }

  std::set<std::string> equationNames;
  for (const auto &equation : problem.equations) {
    if (!equationNames.insert(equation.name).second)
      throw std::runtime_error("constraint equation redeclared: " +
                               equation.name);
    const size_t rank =
        static_cast<size_t>(equation.type.up + equation.type.down);
    if (equation.indices.size() != rank)
      throw std::runtime_error("constraint equation '" + equation.name +
                               "' declares wrong number of indices");
    if (!equation.residual)
      throw std::runtime_error("constraint equation '" + equation.name +
                               "' has no residual");
    for (const auto &index : equation.indices)
      validateSpatialIndex(index);
  }

  std::set<std::string> boundaryRegions;
  for (const auto &boundary : problem.boundaries) {
    if (!boundaryRegions.insert(boundary.region).second)
      throw std::runtime_error("constraint boundary redeclared: " +
                               boundary.region);
    if (boundary.conditions.empty())
      throw std::runtime_error("constraint boundary '" + boundary.region +
                               "' has no conditions");
  }

  if (problem.solve.nonlinear != "newton")
    throw std::runtime_error(
        "constraint solver currently supports nonlinear = newton");
  if (problem.solve.linear != "direct" && problem.solve.linear != "gmres")
    throw std::runtime_error(
        "constraint solver linear method must be direct or gmres");
  if (problem.solve.tolerance <= 0.0)
    throw std::runtime_error("constraint solver tolerance must be > 0");
  if (problem.solve.maxIterations <= 0)
    throw std::runtime_error("constraint solver max_iterations must be > 0");
}

} // namespace tensorium

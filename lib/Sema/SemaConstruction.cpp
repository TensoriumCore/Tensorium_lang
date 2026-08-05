#include "tensorium/Sema/Sema.hpp"

#include <stdexcept>
#include <string>
#include <unordered_set>

namespace tensorium {
namespace {
bool isScalarDesc(const TensorTypeDesc &desc) {
  return desc.up == 0 && desc.down == 0 && desc.kind == TensorKind::Scalar;
}

constexpr const char *kErrMissingSimulationBlock =
    "E1001: missing simulation block in executable mode";

constexpr const char *kWarnMissingSimulationBlock =
    "W1001: missing simulation block in symbolic mode";

constexpr const char *kWarnInverseMetricMissing =
    "W1002: inverse_metric field is missing while metrics are declared";

constexpr const char *kWarnMetricMissing =
    "W1003: metric field is missing while inverse_metric fields are declared";
} // namespace

SemanticAnalyzer::SemanticAnalyzer(const Program &p, CompilationMode m)
    : prog(p), mode(m) {
  for (const auto &paramName : prog.params) {
    if (!params.insert(paramName).second) {
      throw std::runtime_error("Parameter redeclared: " + paramName);
    }
  }

  for (const auto &ext : prog.externs) {
    if (!externSignatures.emplace(ext.name, &ext).second) {
      throw std::runtime_error("Extern function redeclared: " + ext.name);
    }
    if (mode == CompilationMode::Executable && !isScalarDesc(ext.returnType)) {
      throw std::runtime_error("executable mode extern '" + ext.name +
                               "' must return scalar");
    }
  }

  std::unordered_set<std::string> metricNames;
  for (const auto &metric : prog.metrics) {
    if (params.count(metric.name)) {
      throw std::runtime_error("Name collision: parameter '" + metric.name +
                               "' conflicts with metric '" + metric.name +
                               "'");
    }
    if (!metricNames.insert(metric.name).second) {
      throw std::runtime_error("Metric redeclared: " + metric.name);
    }
  }

  for (const auto &f : prog.fields) {
    if (params.count(f.name)) {
      throw std::runtime_error("Name collision: parameter '" + f.name +
                               "' conflicts with field '" + f.name + "'");
    }
    if (metricNames.count(f.name)) {
      throw std::runtime_error("Name collision: field '" + f.name +
                               "' conflicts with metric '" + f.name + "'");
    }
    if (fields.count(f.name))
      throw std::runtime_error("Field redeclared: " + f.name);
    enforceMetricFieldRules(f);
    if ((f.up == 1 && f.down == 2) ||
        ((f.up + f.down) == 3 &&
         (f.name == "Gamma" || f.name == "GammaU" || f.name == "Christoffel"))) {
      hasConnectionTensor = true;
    }
    fields[f.name] = &f;
  }

  if (metricFieldCount > 0 && inverseMetricFieldCount == 0) {
    warnings.push_back(kWarnInverseMetricMissing);
  }
  if (inverseMetricFieldCount > 0 && metricFieldCount == 0) {
    warnings.push_back(kWarnMetricMissing);
  }

  for (const auto &m : prog.metrics) {
    for (const auto &entry : m.entries) {
      if (entry.lhs.indices.empty())
        metricScalarLocals[entry.lhs.base] =
            TensorTypeDesc{TensorKind::Scalar, 0, 0};
    }
  }
  for (const auto &m : prog.metrics) {
    if (fields.count(m.name)) {
      throw std::runtime_error("Name collision: metric '" + m.name +
                               "' conflicts with existing field '" + m.name +
                               "'");
    }
    FieldDecl fd;
    fd.kind = TensorKind::CovTensor2;
    fd.name = m.name;
    fd.up = 0;
    fd.down = 2;

    fd.isMetric = true;

    syntheticMetricFields.push_back(fd);
    fields[m.name] = &syntheticMetricFields.back();
  }

  if (prog.initialData && prog.initialData->hasConstraintProblem) {
    for (const auto &unknown : prog.initialData->constraintProblem.unknowns) {
      if (params.count(unknown.name) || fields.count(unknown.name) ||
          metricNames.count(unknown.name)) {
        throw std::runtime_error("Name collision for constraint unknown '" +
                                 unknown.name + "'");
      }
      if (!unknowns.emplace(unknown.name, &unknown).second) {
        throw std::runtime_error("Constraint unknown redeclared: " +
                                 unknown.name);
      }
    }
  }

  const bool constraintOnly =
      prog.initialData &&
      (prog.initialData->hasConstraintProblem ||
       prog.initialData->hasSpectralProblem) &&
      !prog.initialData->hasMetric4 && !prog.initialData->hasDecomposed &&
      prog.evolutions.empty();
  if (!prog.simulation) {
    simulationMissing = true;
    if (!constraintOnly && mode == CompilationMode::Executable) {
      throw std::runtime_error(
          std::string(kErrMissingSimulationBlock) +
          ". Add `simulation { dimension = <N> resolution = [...] time { dt = ... "
          "integrator = ... } spatial { scheme = ... derivative = ... order = ... } }` "
          "or use --symbolic.");
    }
    if (!constraintOnly) {
      warnings.push_back(std::string(kWarnMissingSimulationBlock) +
                         "; proceeding without simulation metadata");
    }
  } else {
    validateSimulation(*prog.simulation);
  }

  if (prog.initialData) {
    validateInitialData(*prog.initialData);
  }
}

} // namespace tensorium

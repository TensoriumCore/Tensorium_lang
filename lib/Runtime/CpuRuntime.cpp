
#include "tensorium/Runtime/CpuRuntime.hpp"
#include "tensorium/Runtime/Eval.hpp"
#include <stdexcept>

namespace tensorium::runtime {

static void require(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error("runtime: " + msg);
}

CpuState1D initState1D(const backend::ModuleIR& mod,
                       double initScalar,
                       double initAlpha) {
  require(mod.simulation.has_value(), "simulation required");
  require(mod.simulation->dimension == 1, "only dimension=1 supported");

  CpuState1D st;
  st.n = static_cast<std::size_t>(mod.simulation->resolution.at(0));

  for (const auto& f : mod.fields) {
    require(f.up == 0 && f.down == 0, "only scalar fields supported (got " + f.name + ")");
    st.fields[f.name] = std::vector<double>(st.n, initScalar);
  }

  if (st.fields.count("alpha"))
    std::fill(st.fields["alpha"].begin(), st.fields["alpha"].end(), initAlpha);

  return st;
}

void runEuler1D(const backend::ModuleIR& mod, CpuState1D& st, const RunOptions& opt) {
  require(mod.simulation.has_value(), "simulation required");
  const double dt = mod.simulation->time.dt;

  require(mod.simulation->time.integrator == backend::TimeIntegrator::Euler,
          "only euler integrator supported");

  require(mod.evolutions.size() == 1, "runtime expects exactly 1 evolution block for now");

  const auto& evo = mod.evolutions[0];

  std::unordered_map<std::string, std::vector<double>> next = st.fields;

  for (std::size_t step = 0; step < opt.steps; ++step) {
    for (std::size_t i = 0; i < st.n; ++i) {
      ScalarEnv env;
      env.params = st.params;
      for (auto& kv : st.fields) {
        env.fieldPtr[kv.first] = &kv.second[i];
      }

      for (const auto& eq : evo.equations) {
        require(eq.indices.empty(),
                "tensor LHS not supported yet (field " + eq.fieldName + ")");

        auto it = st.fields.find(eq.fieldName);
        require(it != st.fields.end(), "unknown field on LHS: " + eq.fieldName);

        const double rhs = evalScalar(eq.rhs.get(), env);
        next[eq.fieldName][i] = it->second[i] + dt * rhs;
      }
    }

    st.fields.swap(next);
  }
}

} // namespace tensorium::runtime

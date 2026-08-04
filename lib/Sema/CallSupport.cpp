#include "tensorium/Sema/CallSupport.hpp"
#include "tensorium/Core/IndexSet.h"

namespace tensorium {

bool isExecutableBuiltin(std::string_view name) {
  if (name == "sin" || name == "cos" || name == "sqrt" || name == "exp")
    return true;
  if (name == "contract")
    return true;
  if (name == "christoffel")
    return true;
  if (name == "trace")
    return true;
  if (name == "gradient" || name == "grad")
    return true;
  if (name == "divergence" || name == "div")
    return true;
  if (name == "laplacian")
    return true;
  if (name == "covariant_derivative")
    return true;
  if (name.size() == 7 && name.rfind("nabla_", 0) == 0 &&
      core::isSpatialIndexChar(name[6]))
    return true;
  if (name.size() == 7 && name.rfind("nabla^", 0) == 0 &&
      core::isSpatialIndexChar(name[6]))
    return true;
  if (name.size() == 3 && name[0] == 'd' && name[1] == '_' &&
      core::isSpatialIndexChar(name[2]))
    return true;
  return false;
}

bool isRadialConstraintBuiltin(std::string_view name) {
  return name == "radial_derivative" ||
         name == "radial_conformal_vector_laplacian";
}

} // namespace tensorium

#include "tensorium/Sema/CallSupport.hpp"
#include "tensorium/Core/IndexSet.h"

namespace tensorium {

bool isExecutableBuiltin(std::string_view name) {
  if (name == "contract")
    return true;
  if (name.size() == 3 && name[0] == 'd' && name[1] == '_' &&
      core::isSpatialIndexChar(name[2]))
    return true;
  return false;
}

} // namespace tensorium

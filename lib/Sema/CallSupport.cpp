#include "tensorium/Sema/CallSupport.hpp"

namespace tensorium {

static bool isSpatialIndexChar(char c) {
  switch (c) {
  case 'i':
  case 'j':
  case 'k':
  case 'l':
  case 'm':
  case 'n':
    return true;
  default:
    return false;
  }
}

bool isExecutableBuiltin(std::string_view name) {
  if (name == "contract")
    return true;
  if (name.size() == 3 && name[0] == 'd' && name[1] == '_' &&
      isSpatialIndexChar(name[2]))
    return true;
  return false;
}

} // namespace tensorium

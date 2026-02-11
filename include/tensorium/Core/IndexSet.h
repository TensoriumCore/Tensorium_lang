#pragma once

#include <array>
#include <string_view>

namespace tensorium::core {

inline constexpr std::array<char, 6> kTensorIndices = {'i', 'j', 'k',
                                                        'l', 'm', 'n'};

inline constexpr bool isTensorIndexChar(char c) {
  for (char allowed : kTensorIndices) {
    if (c == allowed)
      return true;
  }
  return false;
}

inline bool isTensorIndexName(std::string_view name) {
  return name.size() == 1 && isTensorIndexChar(name.front());
}

inline constexpr bool isSpatialIndexChar(char c) {
  return isTensorIndexChar(c);
}

inline bool isSpatialIndexName(std::string_view name) {
  return isTensorIndexName(name);
}

} // namespace tensorium::core

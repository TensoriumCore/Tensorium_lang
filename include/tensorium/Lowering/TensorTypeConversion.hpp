#pragma once

#include "tensorium/Core/TensorTypes.hpp"
#include "tensorium/IR/DomainIR.hpp"

namespace tensorium::lowering {

inline ir::TensorType makeTensorType(int up, int down) {
  ir::TensorType out;
  out.up = up;
  out.down = down;
  return out;
}

inline ir::TensorType lowerTensorType(const TensorTypeDesc &desc) {
  return makeTensorType(desc.up, desc.down);
}

inline backend::FieldKind lowerFieldKind(TensorKind kind) {
  switch (kind) {
  case TensorKind::Scalar:
    return backend::FieldKind::Scalar;
  case TensorKind::Vector:
    return backend::FieldKind::Vector;
  case TensorKind::Covector:
    return backend::FieldKind::Covector;
  case TensorKind::CovTensor2:
    return backend::FieldKind::CovTensor2;
  case TensorKind::ConTensor2:
    return backend::FieldKind::ConTensor2;
  case TensorKind::CovTensor3:
    return backend::FieldKind::CovTensor3;
  case TensorKind::ConTensor3:
    return backend::FieldKind::ConTensor3;
  case TensorKind::CovTensor4:
    return backend::FieldKind::CovTensor4;
  case TensorKind::ConTensor4:
    return backend::FieldKind::ConTensor4;
  case TensorKind::MixedTensor:
    return backend::FieldKind::MixedTensor;
  case TensorKind::Metric:
    return backend::FieldKind::CovTensor2;
  case TensorKind::InverseMetric:
    return backend::FieldKind::ConTensor2;
  }
  return backend::FieldKind::Scalar;
}

} // namespace tensorium::lowering

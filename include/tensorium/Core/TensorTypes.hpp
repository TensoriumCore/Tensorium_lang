#pragma once

namespace tensorium {

enum class TensorKind {
  Scalar,
  Vector,
  Covector,
  CovTensor2,
  ConTensor2,
  CovTensor3,
  ConTensor3,
  ConTensor4,
  CovTensor4,
  MixedTensor,
  Metric,
  InverseMetric
};

struct TensorTypeDesc {
  TensorKind kind = TensorKind::Scalar;
  int up = 0;
  int down = 0;
};

namespace core {

inline TensorKind deduceTensorKind(int up, int down) {
  if (up == 0 && down == 0)
    return TensorKind::Scalar;
  if (up == 1 && down == 0)
    return TensorKind::Vector;
  if (up == 0 && down == 1)
    return TensorKind::Covector;
  if (up == 0 && down == 2)
    return TensorKind::CovTensor2;
  if (up == 2 && down == 0)
    return TensorKind::ConTensor2;
  if (up == 0 && down == 3)
    return TensorKind::CovTensor3;
  if (up == 3 && down == 0)
    return TensorKind::ConTensor3;
  if (up == 0 && down == 4)
    return TensorKind::CovTensor4;
  if (up == 4 && down == 0)
    return TensorKind::ConTensor4;
  return TensorKind::MixedTensor;
}

inline TensorTypeDesc makeTensorTypeDesc(int up, int down) {
  return TensorTypeDesc{deduceTensorKind(up, down), up, down};
}

inline bool isScalarTensorType(const TensorTypeDesc &desc) {
  return desc.up == 0 && desc.down == 0 && desc.kind == TensorKind::Scalar;
}

inline int declaredContravariantCount(TensorKind kind, int mixedUp = 0) {
  switch (kind) {
  case TensorKind::Scalar:
  case TensorKind::Covector:
  case TensorKind::CovTensor2:
  case TensorKind::CovTensor3:
  case TensorKind::CovTensor4:
  case TensorKind::Metric:
    return 0;
  case TensorKind::Vector:
    return 1;
  case TensorKind::ConTensor2:
  case TensorKind::InverseMetric:
    return 2;
  case TensorKind::ConTensor3:
    return 3;
  case TensorKind::ConTensor4:
    return 4;
  case TensorKind::MixedTensor:
    return mixedUp;
  }
  return 0;
}

} // namespace core
} // namespace tensorium

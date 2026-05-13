#ifndef TENSORIUM_DIALECT_TD
#define TENSORIUM_DIALECT_TD

include "mlir/IR/DialectBase.td"

def Tensorium_Dialect : Dialect {
  let name = "tensorium";
  let cppNamespace = "::tensorium::mlir";
  let useDefaultTypePrinterParser = 1;
}

#endif // TENSORIUM_DIALECT_TD

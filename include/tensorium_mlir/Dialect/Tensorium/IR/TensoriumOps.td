include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def Tensorium_Dialect : Dialect {
  let name = "tensorium";
  let cppNamespace = "::tensorium::mlir";
}

class Tensorium_Op<string mnemonic, list<Trait> traits = []>
    : Op<Tensorium_Dialect, mnemonic, traits>;

def Tensorium_IndexOp : Tensorium_Op<"index", [
  SameOperandsAndResultType
]> {
  let arguments = (ins
    AnyType:$field,
    ArrayAttr:$indices
  );

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = "$field `[` $indices `]` `:` type($result) attr-dict";
  let hasVerifier = 1;
}

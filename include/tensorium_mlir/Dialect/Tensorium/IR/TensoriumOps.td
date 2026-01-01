include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tensorium_Dialect : Dialect {
  let name = "tensorium";
  let cppNamespace = "::tensorium::mlir";
}

class Tensorium_Op<string mnemonic>
    : Op<Tensorium_Dialect, mnemonic>;

def Tensorium_IndexOp : Tensorium_Op<"index"> {
  let summary = "Symbolic tensor indexing";

  let arguments = (ins
    AnyType:$field,
    ArrayAttr:$indices
  );

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat =
    "$field `[` $indices `]` attr-dict `:` type($field) `->` type($result)";

  let hasVerifier = 1;
}

def Tensorium_ConstOp : Tensorium_Op<"const"> {
  let arguments = (ins F64Attr:$value);
  let results = (outs F64:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Tensorium_RefOp : Tensorium_Op<"ref"> {
  let arguments = (ins 
    AnyType:$source, 
    StrAttr:$kind, 
    OptionalAttr<ArrayAttr>:$indices,
    OptionalAttr<I64ArrayAttr>:$offsets 
  );
  let results = (outs F64:$result);
  
  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];
}

def Tensorium_AddOp : Tensorium_Op<"add"> {
  let traits = [Pure];

  let arguments = (ins F64:$lhs, F64:$rhs);
  let results   = (outs F64:$res);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

def Tensorium_SubOp : Tensorium_Op<"sub"> {
  let traits = [Pure];

  let arguments = (ins F64:$lhs, F64:$rhs);
  let results   = (outs F64:$res);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

def Tensorium_MulOp : Tensorium_Op<"mul"> {
  let traits = [Pure];

  let arguments = (ins F64:$lhs, F64:$rhs);
  let results   = (outs F64:$res);
}

def Tensorium_DerivOp : Tensorium_Op<"deriv"> {
  let traits = [Pure];

  let arguments = (ins F64:$in);
  let results   = (outs F64:$out);
  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tensorium_ContractOp : Tensorium_Op<"contract"> {
  let traits = [Pure];

  let arguments = (ins F64:$in);
  let results   = (outs F64:$out);
  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
}

def Tensorium_DtAssignOp : Tensorium_Op<"dt_assign"> {

  let arguments = (ins AnyType:$field, F64:$rhs, ArrayAttr:$indices);
  let results = (outs);
  let assemblyFormat =
    "$field `,` $rhs attr-dict `:` type($field) `,` type($rhs)";
}



def Tensorium_EinsumOp : Tensorium_Op<"einsum"> {
  let summary = "Einstein summation (tensor contraction)";

  let traits = [Pure];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results   = (outs AnyType:$result);

  let hasCustomAssemblyFormat = 1;
}




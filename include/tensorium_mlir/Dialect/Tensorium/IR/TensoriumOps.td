include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tensorium_FieldType : Type<CPred<"::llvm::isa<::tensorium::mlir::FieldType>($_self)">,
                              "Tensorium field type">;

def Tensorium_Dialect : Dialect {
  let name = "tensorium";
  let cppNamespace = "::tensorium::mlir";
}

class Tensorium_Op<string mnemonic>
    : Op<Tensorium_Dialect, mnemonic>;

def Tensorium_IndexOp : Tensorium_Op<"index"> {
  let summary = "Symbolic tensor indexing";

  let arguments = (ins Tensorium_FieldType:$field, ArrayAttr:$indices);

  let results = (outs Tensorium_FieldType:$result);

  let assemblyFormat =
    "$field `[` $indices `]` attr-dict `:` type($field) `->` type($result)";

  let hasVerifier = 1;
}

def Tensorium_ConstOp : Tensorium_Op<"const"> {
  let arguments = (ins F64Attr:$value);
  let results = (outs Tensorium_FieldType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Tensorium_RefOp : Tensorium_Op<"ref"> {
  let arguments = (ins Tensorium_FieldType:$source, StrAttr:$kind,
                   OptionalAttr<ArrayAttr>:$indices,
                   OptionalAttr<I64ArrayAttr>:$offsets);
  let results = (outs Tensorium_FieldType:$result);
  let hasVerifier = 1;
  
  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];
}

def Tensorium_ParamOp : Tensorium_Op<"param"> {
  let arguments = (ins StrAttr:$name);
  let results = (outs Tensorium_FieldType:$result);
  let hasVerifier = 1;
}

def Tensorium_CoordOp : Tensorium_Op<"coord"> {
  let arguments = (ins StrAttr:$name);
  let results = (outs Tensorium_FieldType:$result);
  let hasVerifier = 1;
}

def Tensorium_AddOp : Tensorium_Op<"add"> {
  let traits = [Pure];

  let arguments = (ins Tensorium_FieldType:$lhs, Tensorium_FieldType:$rhs);
  let results   = (outs Tensorium_FieldType:$res);
  let hasVerifier = 1;
}

def Tensorium_SubOp : Tensorium_Op<"sub"> {
  let traits = [Pure];

  let arguments = (ins Tensorium_FieldType:$lhs, Tensorium_FieldType:$rhs);
  let results   = (outs Tensorium_FieldType:$res);
  let hasVerifier = 1;
}

def Tensorium_MulOp : Tensorium_Op<"mul"> {
  let traits = [Pure];

  let arguments = (ins Tensorium_FieldType:$lhs, Tensorium_FieldType:$rhs);
  let results   = (outs Tensorium_FieldType:$res);
  let hasVerifier = 1;
}
def Tensorium_DivOp : Tensorium_Op<"div"> {
  let traits = [Pure];

  let arguments = (ins Tensorium_FieldType:$lhs, Tensorium_FieldType:$rhs);
  let results   = (outs Tensorium_FieldType:$res);
  let hasVerifier = 1;
}

def Tensorium_SinOp : Tensorium_Op<"sin"> {
  let traits = [Pure];

  let arguments = (ins Tensorium_FieldType:$in);
  let results = (outs Tensorium_FieldType:$out);
  let hasVerifier = 1;
}

def Tensorium_SqrtOp : Tensorium_Op<"sqrt"> {
  let traits = [Pure];

  let arguments = (ins Tensorium_FieldType:$in);
  let results = (outs Tensorium_FieldType:$out);
  let hasVerifier = 1;
}

def Tensorium_DerivOp : Tensorium_Op<"deriv"> {
  let traits = [Pure];

  let arguments = (ins Tensorium_FieldType:$in);
  let results   = (outs Tensorium_FieldType:$out);
  let hasVerifier = 1;
}

def Tensorium_ContractOp : Tensorium_Op<"contract"> {
  let traits = [Pure];

  let arguments = (ins Tensorium_FieldType:$in);
  let results   = (outs Tensorium_FieldType:$out);
  let hasVerifier = 1;
}

def Tensorium_PromoteOp : Tensorium_Op<"promote"> {
  let summary = "Promote scalar tensor value to specified field type";

  let arguments = (ins Tensorium_FieldType:$in);
  let results   = (outs Tensorium_FieldType:$out);
  let hasVerifier = 1;
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";
}

def Tensorium_DtAssignOp : Tensorium_Op<"dt_assign"> {

  let arguments = (ins Tensorium_FieldType:$field, Tensorium_FieldType:$rhs,
                   ArrayAttr:$indices);
  let results = (outs);
  let hasVerifier = 1;
}

def Tensorium_Metric4Op : Tensorium_Op<"metric4"> {
  let summary = "Carry explicit 4D metric initial data";
  let arguments = (ins Variadic<Tensorium_FieldType>:$components,
                   StrAttr:$name, StrAttr:$coord_system,
                   ArrayAttr:$indices,
                   BoolAttr:$enforce_symmetry);
  let results = (outs Tensorium_FieldType:$metric);
  let hasVerifier = 1;
}

def Tensorium_Split3P1Op : Tensorium_Op<"split3p1"> {
  let summary = "Split spacetime metric into alpha, beta, gamma, gammaU";
  let arguments = (ins Tensorium_FieldType:$metric4,
                   Tensorium_FieldType:$alpha_in,
                   Tensorium_FieldType:$beta_in,
                   Tensorium_FieldType:$gamma_in,
                   Tensorium_FieldType:$gammaU_in);
  let results = (outs Tensorium_FieldType:$alpha,
                      Tensorium_FieldType:$beta,
                      Tensorium_FieldType:$gamma,
                      Tensorium_FieldType:$gammaU);
  let hasVerifier = 1;
}

def Tensorium_BuildCovectorOp : Tensorium_Op<"build_covector"> {
  let arguments = (ins Variadic<Tensorium_FieldType>:$components);
  let results = (outs Tensorium_FieldType:$out);
  let hasVerifier = 1;
}

def Tensorium_BuildCovTensor2Op : Tensorium_Op<"build_cov_tensor2"> {
  let arguments = (ins Variadic<Tensorium_FieldType>:$components);
  let results = (outs Tensorium_FieldType:$out);
  let hasVerifier = 1;
}

def Tensorium_BuildConTensor2Op : Tensorium_Op<"build_con_tensor2"> {
  let arguments = (ins Variadic<Tensorium_FieldType>:$components);
  let results = (outs Tensorium_FieldType:$out);
  let hasVerifier = 1;
}



def Tensorium_EinsumOp : Tensorium_Op<"einsum"> {
  let summary = "Einstein summation (tensor contraction)";

  let traits = [Pure];
  let arguments = (ins Variadic<Tensorium_FieldType>:$inputs);
  let results   = (outs Tensorium_FieldType:$result);

  let hasCustomAssemblyFormat = 1;
}

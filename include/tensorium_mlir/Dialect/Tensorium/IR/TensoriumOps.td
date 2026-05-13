include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.td"

class Tensorium_Op<string mnemonic>
    : Op<Tensorium_Dialect, mnemonic>;

def Tensorium_NonEmptyStrAttr : ConfinedAttr<StrAttr, [
  AttrConstraint<
    CPred<"!::llvm::cast<::mlir::StringAttr>($_self).getValue().empty()">,
    "whose value is non-empty">
]>;

def Tensorium_CoordSystemAttr : ConfinedAttr<StrAttr, [
  AttrConstraint<
    CPred<"::llvm::cast<::mlir::StringAttr>($_self).getValue() == \"cartesian\" || "
          "::llvm::cast<::mlir::StringAttr>($_self).getValue() == \"spherical\" || "
          "::llvm::cast<::mlir::StringAttr>($_self).getValue() == \"cylindrical\"">,
    "whose value is cartesian, spherical, or cylindrical">
]>;

def Tensorium_IndexArrayAttr
    : TypedArrayAttrBase<StrAttr, "string array attribute"> {
  let constBuilderCall = "$_builder.getStrArrayAttr($0)";
}
def Tensorium_IndexPairAttr
    : ConfinedAttr<Tensorium_IndexArrayAttr, [ArrayCount<2>]>;

class Tensorium_ComponentCount<int n> : PredOpTrait<
  "components have expected count",
  CPred<"$_op.getNumOperands() == " # n>>;

def Tensorium_MulRankMatches : PredOpTrait<
  "result rank equals operand rank sum",
  CPred<"::llvm::cast<::tensorium::mlir::FieldType>($res.getType()).getRank() == "
        "::llvm::cast<::tensorium::mlir::FieldType>($lhs.getType()).getRank() + "
        "::llvm::cast<::tensorium::mlir::FieldType>($rhs.getType()).getRank()">>;

def Tensorium_DerivRankMatches : PredOpTrait<
  "derivative adds one covariant index",
  CPred<"::llvm::cast<::tensorium::mlir::FieldType>($out.getType()).getRank() == "
        "::llvm::cast<::tensorium::mlir::FieldType>($in.getType()).getRank() + 1">>;

def Tensorium_ContractRankDoesNotGrow : PredOpTrait<
  "contract result cannot have more indices than input",
  CPred<"::llvm::cast<::tensorium::mlir::FieldType>($out.getType()).getRank() <= "
        "::llvm::cast<::tensorium::mlir::FieldType>($in.getType()).getRank()">>;

def Tensorium_IndexOp : Tensorium_Op<"index"> {
  let summary = "Symbolic tensor indexing";

  let arguments = (ins Tensorium_FieldType:$field,
                       Tensorium_IndexArrayAttr:$indices);

  let results = (outs Tensorium_FieldType:$result);

  let assemblyFormat =
    "$field `[` $indices `]` attr-dict `:` type($field) `->` type($result)";

  let hasVerifier = 1;
}

def Tensorium_ConstOp : Tensorium_Op<"const"> {
  let traits = [Pure];
  let arguments = (ins F64Attr:$value);
  let results = (outs Tensorium_FieldType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Tensorium_RefOp : Tensorium_Op<"ref"> {
  let traits = [AllTypesMatch<["source", "result"]>];
  let arguments = (ins Tensorium_FieldType:$source, StrAttr:$kind,
                   OptionalAttr<Tensorium_IndexArrayAttr>:$indices,
                   OptionalAttr<I64ArrayAttr>:$offsets);
  let results = (outs Tensorium_FieldType:$result);
  
  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];
}

def Tensorium_ParamOp : Tensorium_Op<"param"> {
  let traits = [Pure];
  let arguments = (ins Tensorium_NonEmptyStrAttr:$name);
  let results = (outs Tensorium_ScalarFieldType:$result);
}

def Tensorium_CoordOp : Tensorium_Op<"coord"> {
  let traits = [Pure];
  let arguments = (ins Tensorium_NonEmptyStrAttr:$name);
  let results = (outs Tensorium_ScalarFieldType:$result);
}

def Tensorium_AddOp : Tensorium_Op<"add"> {
  let traits = [Pure, AllTypesMatch<["lhs", "rhs", "res"]>];

  let arguments = (ins Tensorium_FieldType:$lhs, Tensorium_FieldType:$rhs);
  let results   = (outs Tensorium_FieldType:$res);
}

def Tensorium_SubOp : Tensorium_Op<"sub"> {
  let traits = [Pure, AllTypesMatch<["lhs", "rhs", "res"]>];

  let arguments = (ins Tensorium_FieldType:$lhs, Tensorium_FieldType:$rhs);
  let results   = (outs Tensorium_FieldType:$res);
}

def Tensorium_MulOp : Tensorium_Op<"mul"> {
  let traits = [Pure, Tensorium_MulRankMatches];

  let arguments = (ins Tensorium_FieldType:$lhs, Tensorium_FieldType:$rhs);
  let results   = (outs Tensorium_FieldType:$res);
}
def Tensorium_DivOp : Tensorium_Op<"div"> {
  let traits = [Pure, AllTypesMatch<["lhs", "res"]>];

  let arguments = (ins Tensorium_FieldType:$lhs, Tensorium_ScalarFieldType:$rhs);
  let results   = (outs Tensorium_FieldType:$res);
}

def Tensorium_SinOp : Tensorium_Op<"sin"> {
  let traits = [Pure, AllTypesMatch<["in", "out"]>];

  let arguments = (ins Tensorium_ScalarFieldType:$in);
  let results = (outs Tensorium_ScalarFieldType:$out);
}

def Tensorium_SqrtOp : Tensorium_Op<"sqrt"> {
  let traits = [Pure, AllTypesMatch<["in", "out"]>];

  let arguments = (ins Tensorium_ScalarFieldType:$in);
  let results = (outs Tensorium_ScalarFieldType:$out);
}

def Tensorium_DerivOp : Tensorium_Op<"deriv"> {
  let traits = [Pure, Tensorium_DerivRankMatches];

  let arguments = (ins Tensorium_FieldType:$in);
  let results   = (outs Tensorium_FieldType:$out);
}

def Tensorium_ContractOp : Tensorium_Op<"contract"> {
  let traits = [Pure, Tensorium_ContractRankDoesNotGrow];

  let arguments = (ins Tensorium_FieldType:$in);
  let results   = (outs Tensorium_FieldType:$out);
}

def Tensorium_PromoteOp : Tensorium_Op<"promote"> {
  let summary = "Promote scalar tensor value to specified field type";

  let arguments = (ins Tensorium_ScalarFieldType:$in);
  let results   = (outs Tensorium_FieldType:$out);
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";
}

def Tensorium_ExternCallOp : Tensorium_Op<"extern_call"> {
  let summary = "Pure scalar external function call";
  let traits = [Pure];

  let arguments = (ins Tensorium_NonEmptyStrAttr:$callee,
                       Variadic<Tensorium_ScalarFieldType>:$args);
  let results = (outs Tensorium_ScalarFieldType:$result);
  let assemblyFormat =
    "$callee `(` $args `)` attr-dict `:` functional-type($args, results)";
}

def Tensorium_DtAssignOp : Tensorium_Op<"dt_assign"> {
  let traits = [AllTypesMatch<["field", "rhs"]>];

  let arguments = (ins Tensorium_FieldType:$field, Tensorium_FieldType:$rhs,
                   Tensorium_IndexArrayAttr:$indices);
  let results = (outs);
}

def Tensorium_AssignOp : Tensorium_Op<"assign"> {
  let traits = [AllTypesMatch<["field", "rhs"]>];

  let arguments = (ins Tensorium_FieldType:$field, Tensorium_FieldType:$rhs,
                   Tensorium_IndexArrayAttr:$indices);
  let results = (outs);
}

def Tensorium_Metric4Op : Tensorium_Op<"metric4"> {
  let summary = "Carry explicit 4D metric initial data";
  let traits = [Tensorium_ComponentCount<16>];
  let arguments = (ins Variadic<Tensorium_ScalarFieldType>:$components,
                   Tensorium_NonEmptyStrAttr:$name,
                   Tensorium_CoordSystemAttr:$coord_system,
                   Tensorium_IndexPairAttr:$indices,
                   BoolAttr:$enforce_symmetry);
  let results = (outs Tensorium_CovTensor2FieldType:$metric);
}

def Tensorium_Decompose3P1FromMetricOp : Tensorium_Op<"decompose3p1_from_metric"> {
  let summary = "Decompose 4D metric into alpha, beta, gamma, gammaU";
  let arguments = (ins Tensorium_CovTensor2FieldType:$metric4);
  let results = (outs Tensorium_ScalarFieldType:$alpha,
                      Tensorium_CovectorFieldType:$beta,
                      Tensorium_CovTensor2FieldType:$gamma,
                      Tensorium_ConTensor2FieldType:$gammaU);
}

def Tensorium_Init3P1Op : Tensorium_Op<"init3p1"> {
  let summary = "Carry explicit 3+1 initial data values";
  let traits = [
    AllTypesMatch<["alpha_in", "alpha"]>,
    AllTypesMatch<["beta_in", "beta"]>,
    AllTypesMatch<["gamma_in", "gamma"]>,
    AllTypesMatch<["gammaU_in", "gammaU"]>
  ];
  let arguments = (ins Tensorium_ScalarFieldType:$alpha_in,
                   Tensorium_CovectorFieldType:$beta_in,
                   Tensorium_CovTensor2FieldType:$gamma_in,
                   Tensorium_ConTensor2FieldType:$gammaU_in);
  let results = (outs Tensorium_ScalarFieldType:$alpha,
                      Tensorium_CovectorFieldType:$beta,
                      Tensorium_CovTensor2FieldType:$gamma,
                      Tensorium_ConTensor2FieldType:$gammaU);
}

def Tensorium_BuildCovectorOp : Tensorium_Op<"build_covector"> {
  let traits = [Tensorium_ComponentCount<3>];
  let arguments = (ins Variadic<Tensorium_ScalarFieldType>:$components);
  let results = (outs Tensorium_CovectorFieldType:$out);
}

def Tensorium_BuildCovTensor2Op : Tensorium_Op<"build_cov_tensor2"> {
  let traits = [Tensorium_ComponentCount<9>];
  let arguments = (ins Variadic<Tensorium_ScalarFieldType>:$components);
  let results = (outs Tensorium_CovTensor2FieldType:$out);
}

def Tensorium_BuildConTensor2Op : Tensorium_Op<"build_con_tensor2"> {
  let traits = [Tensorium_ComponentCount<9>];
  let arguments = (ins Variadic<Tensorium_ScalarFieldType>:$components);
  let results = (outs Tensorium_ConTensor2FieldType:$out);
}



def Tensorium_EinsumOp : Tensorium_Op<"einsum"> {
  let summary = "Einstein summation (tensor contraction)";

  let traits = [Pure];
  let arguments = (ins Variadic<Tensorium_FieldType>:$inputs);
  let results   = (outs Tensorium_FieldType:$result);

  let hasCustomAssemblyFormat = 1;
}

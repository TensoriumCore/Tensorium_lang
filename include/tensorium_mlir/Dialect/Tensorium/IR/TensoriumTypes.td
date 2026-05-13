#ifndef TENSORIUM_TYPES_TD
#define TENSORIUM_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "tensorium_mlir/Dialect/Tensorium/IR/TensoriumDialect.td"

def Tensorium_FieldType : TypeDef<Tensorium_Dialect, "Field"> {
  let summary = "Tensor-valued field type";
  let mnemonic = "field";
  let parameters = (ins "::mlir::Type":$elementType,
                        "unsigned":$up,
                        "unsigned":$down);
  let assemblyFormat = "`<` $elementType `,` $up `,` $down `>`";
  let extraClassDeclaration = [{
    unsigned getRank() const { return getUp() + getDown(); }
    Variance getVariance() const;
  }];
}

class Tensorium_FieldTypeWith<Pred pred, string summary>
    : Type<And<[
          CPred<"::llvm::isa<::tensorium::mlir::FieldType>($_self)">,
          pred
        ]>, summary, "::tensorium::mlir::FieldType">;

def Tensorium_ScalarFieldType : Tensorium_FieldTypeWith<
  CPred<"::llvm::cast<::tensorium::mlir::FieldType>($_self).getRank() == 0">,
  "scalar Tensorium field type">;

def Tensorium_CovectorFieldType : Tensorium_FieldTypeWith<
  CPred<"::llvm::cast<::tensorium::mlir::FieldType>($_self).getUp() == 0 && ::llvm::cast<::tensorium::mlir::FieldType>($_self).getDown() == 1">,
  "covector Tensorium field type">;

def Tensorium_CovTensor2FieldType : Tensorium_FieldTypeWith<
  CPred<"::llvm::cast<::tensorium::mlir::FieldType>($_self).getUp() == 0 && ::llvm::cast<::tensorium::mlir::FieldType>($_self).getDown() == 2">,
  "covariant rank-2 Tensorium field type">;

def Tensorium_ConTensor2FieldType : Tensorium_FieldTypeWith<
  CPred<"::llvm::cast<::tensorium::mlir::FieldType>($_self).getUp() == 2 && ::llvm::cast<::tensorium::mlir::FieldType>($_self).getDown() == 0">,
  "contravariant rank-2 Tensorium field type">;

#endif // TENSORIUM_TYPES_TD

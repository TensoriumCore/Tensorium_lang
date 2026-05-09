module attributes {tensorium.abi.memory_layout = "soa_component_major", tensorium.abi.memref_abi = "strided_memref_rank1_f64", tensorium.abi.version = 1 : i64, tensorium.sim.coords = "cartesian", tensorium.sim.dim = 3 : i64} {
  func.func @tensorium_init_point(%arg0: f64, %arg1: f64, %arg2: f64, %arg3: memref<1xf64>, %arg4: memref<9xf64>, %arg5: memref<9xf64>) attributes {tensorium.abi.coord_names = ["x", "y", "z"], tensorium.abi.kind = "init_point", tensorium.abi.memory_layout = "soa_component_major", tensorium.abi.memref_abi = "strided_memref_rank1_f64", tensorium.abi.output_names = ["alpha", "gamma", "gammaU"], tensorium.abi.param_names = [], tensorium.abi.version = 1 : i64, tensorium.abi.write_arg_indices = [3, 4, 5], tensorium.init.coord_names = ["x", "y", "z"], tensorium.init.param_names = []} {
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 4.000000e+00 : f64
    %cst_0 = arith.constant 3.000000e+00 : f64
    %cst_1 = arith.constant 2.000000e+00 : f64
    %cst_2 = arith.constant 0.000000e+00 : f64
    %cst_3 = arith.constant 1.000000e+00 : f64
    %cst_4 = arith.constant 6.000000e-01 : f64
    %cst_5 = arith.constant -2.000000e-01 : f64
    %cst_6 = arith.constant 4.000000e-01 : f64
    %cst_7 = arith.constant 2.500000e-01 : f64
    memref.store %cst_3, %arg3[%c0] : memref<1xf64>
    memref.store %cst_1, %arg4[%c0] : memref<9xf64>
    memref.store %cst_3, %arg4[%c1] : memref<9xf64>
    memref.store %cst_2, %arg4[%c2] : memref<9xf64>
    memref.store %cst_3, %arg4[%c3] : memref<9xf64>
    memref.store %cst_0, %arg4[%c4] : memref<9xf64>
    memref.store %cst_2, %arg4[%c5] : memref<9xf64>
    memref.store %cst_2, %arg4[%c6] : memref<9xf64>
    memref.store %cst_2, %arg4[%c7] : memref<9xf64>
    memref.store %cst, %arg4[%c8] : memref<9xf64>
    memref.store %cst_4, %arg5[%c0] : memref<9xf64>
    memref.store %cst_5, %arg5[%c1] : memref<9xf64>
    memref.store %cst_2, %arg5[%c2] : memref<9xf64>
    memref.store %cst_5, %arg5[%c3] : memref<9xf64>
    memref.store %cst_6, %arg5[%c4] : memref<9xf64>
    memref.store %cst_2, %arg5[%c5] : memref<9xf64>
    memref.store %cst_2, %arg5[%c6] : memref<9xf64>
    memref.store %cst_2, %arg5[%c7] : memref<9xf64>
    memref.store %cst_7, %arg5[%c8] : memref<9xf64>
    return
  }
  func.func @tensorium_init_grid_affine(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {tensorium.abi.coord_names = ["x", "y", "z"], tensorium.abi.kind = "init_grid_affine", tensorium.abi.memory_layout = "soa_component_major", tensorium.abi.memref_abi = "strided_memref_rank1_f64", tensorium.abi.output_names = ["alpha", "gamma", "gammaU"], tensorium.abi.param_names = [], tensorium.abi.version = 1 : i64, tensorium.abi.write_arg_indices = [3, 4, 5]} {
    %cst = arith.constant 2.500000e-01 : f64
    %cst_0 = arith.constant 4.000000e-01 : f64
    %cst_1 = arith.constant -2.000000e-01 : f64
    %cst_2 = arith.constant 6.000000e-01 : f64
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %cst_3 = arith.constant 4.000000e+00 : f64
    %cst_4 = arith.constant 3.000000e+00 : f64
    %cst_5 = arith.constant 2.000000e+00 : f64
    %cst_6 = arith.constant 1.000000e+00 : f64
    %cst_7 = arith.constant 0.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf64>
    affine.for %arg6 = 0 to %dim {
      memref.store %cst_6, %arg3[%arg6] : memref<?xf64>
      memref.store %cst_5, %arg4[%arg6] : memref<?xf64>
      %0 = arith.addi %dim, %arg6 : index
      memref.store %cst_6, %arg4[%0] : memref<?xf64>
      %1 = arith.muli %dim, %c2 : index
      %2 = arith.addi %1, %arg6 : index
      memref.store %cst_7, %arg4[%2] : memref<?xf64>
      %3 = arith.muli %dim, %c3 : index
      %4 = arith.addi %3, %arg6 : index
      memref.store %cst_6, %arg4[%4] : memref<?xf64>
      %5 = arith.muli %dim, %c4 : index
      %6 = arith.addi %5, %arg6 : index
      memref.store %cst_4, %arg4[%6] : memref<?xf64>
      %7 = arith.muli %dim, %c5 : index
      %8 = arith.addi %7, %arg6 : index
      memref.store %cst_7, %arg4[%8] : memref<?xf64>
      %9 = arith.muli %dim, %c6 : index
      %10 = arith.addi %9, %arg6 : index
      memref.store %cst_7, %arg4[%10] : memref<?xf64>
      %11 = arith.muli %dim, %c7 : index
      %12 = arith.addi %11, %arg6 : index
      memref.store %cst_7, %arg4[%12] : memref<?xf64>
      %13 = arith.muli %dim, %c8 : index
      %14 = arith.addi %13, %arg6 : index
      memref.store %cst_3, %arg4[%14] : memref<?xf64>
      memref.store %cst_2, %arg5[%arg6] : memref<?xf64>
      memref.store %cst_1, %arg5[%0] : memref<?xf64>
      memref.store %cst_7, %arg5[%2] : memref<?xf64>
      memref.store %cst_1, %arg5[%4] : memref<?xf64>
      memref.store %cst_0, %arg5[%6] : memref<?xf64>
      memref.store %cst_7, %arg5[%8] : memref<?xf64>
      memref.store %cst_7, %arg5[%10] : memref<?xf64>
      memref.store %cst_7, %arg5[%12] : memref<?xf64>
      memref.store %cst, %arg5[%14] : memref<?xf64>
    }
    return
  }
  func.func @tensorium_rhs_grid_affine(%arg0: index, %arg1: index, %arg2: index, %arg3: f64, %arg4: f64, %arg5: f64, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>) attributes {tensorium.abi.field_names = ["alpha", "phi", "H", "gamma", "gammaU", "K"], tensorium.abi.kind = "rhs_grid_affine", tensorium.abi.memory_layout = "soa_component_major", tensorium.abi.memref_abi = "strided_memref_rank1_f64", tensorium.abi.output_names = ["H", "K"], tensorium.abi.param_names = [], tensorium.abi.version = 1 : i64, tensorium.abi.write_arg_indices = [8, 11]} {
    %c-1 = arith.constant -1 : index
    %cst = arith.constant 4.000000e+00 : f64
    %cst_0 = arith.constant 2.000000e+00 : f64
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0 = arith.subi %arg0, %c1 : index
    %1 = arith.subi %arg1, %c1 : index
    %2 = arith.subi %arg2, %c1 : index
    affine.for %arg12 = 1 to %0 {
      affine.for %arg13 = 1 to %1 {
        affine.for %arg14 = 1 to %2 {
          %3 = arith.muli %arg0, %arg1 : index
          %4 = arith.muli %3, %arg2 : index
          %5 = arith.muli %arg12, %arg1 : index
          %6 = arith.addi %5, %arg13 : index
          %7 = arith.muli %6, %arg2 : index
          %8 = arith.addi %7, %arg14 : index
          %9 = memref.load %arg10[%8] : memref<?xf64>
          %10 = arith.addi %4, %8 : index
          %11 = memref.load %arg10[%10] : memref<?xf64>
          %12 = arith.muli %4, %c2 : index
          %13 = arith.addi %12, %8 : index
          %14 = memref.load %arg10[%13] : memref<?xf64>
          %15 = arith.muli %4, %c3 : index
          %16 = arith.addi %15, %8 : index
          %17 = memref.load %arg10[%16] : memref<?xf64>
          %18 = arith.muli %4, %c4 : index
          %19 = arith.addi %18, %8 : index
          %20 = memref.load %arg10[%19] : memref<?xf64>
          %21 = arith.muli %4, %c5 : index
          %22 = arith.addi %21, %8 : index
          %23 = memref.load %arg10[%22] : memref<?xf64>
          %24 = arith.muli %4, %c6 : index
          %25 = arith.addi %24, %8 : index
          %26 = memref.load %arg10[%25] : memref<?xf64>
          %27 = arith.muli %4, %c7 : index
          %28 = arith.addi %27, %8 : index
          %29 = memref.load %arg10[%28] : memref<?xf64>
          %30 = arith.muli %4, %c8 : index
          %31 = arith.addi %30, %8 : index
          %32 = memref.load %arg10[%31] : memref<?xf64>
          %33 = memref.load %arg7[%8] : memref<?xf64>
          %34 = arith.addi %arg12, %c1 : index
          %35 = arith.muli %34, %arg1 : index
          %36 = arith.addi %35, %arg13 : index
          %37 = arith.muli %36, %arg2 : index
          %38 = arith.addi %37, %arg14 : index
          %39 = memref.load %arg7[%38] : memref<?xf64>
          %40 = arith.addi %arg12, %c-1 : index
          %41 = arith.muli %40, %arg1 : index
          %42 = arith.addi %41, %arg13 : index
          %43 = arith.muli %42, %arg2 : index
          %44 = arith.addi %43, %arg14 : index
          %45 = memref.load %arg7[%44] : memref<?xf64>
          %46 = arith.mulf %arg3, %arg3 : f64
          %47 = arith.mulf %33, %cst_0 : f64
          %48 = arith.addf %39, %45 : f64
          %49 = arith.subf %48, %47 : f64
          %50 = arith.divf %49, %46 : f64
          %51 = arith.addi %arg13, %c1 : index
          %52 = arith.addi %35, %51 : index
          %53 = arith.muli %52, %arg2 : index
          %54 = arith.addi %53, %arg14 : index
          %55 = memref.load %arg7[%54] : memref<?xf64>
          %56 = arith.addi %41, %51 : index
          %57 = arith.muli %56, %arg2 : index
          %58 = arith.addi %57, %arg14 : index
          %59 = memref.load %arg7[%58] : memref<?xf64>
          %60 = arith.addi %arg13, %c-1 : index
          %61 = arith.addi %35, %60 : index
          %62 = arith.muli %61, %arg2 : index
          %63 = arith.addi %62, %arg14 : index
          %64 = memref.load %arg7[%63] : memref<?xf64>
          %65 = arith.addi %41, %60 : index
          %66 = arith.muli %65, %arg2 : index
          %67 = arith.addi %66, %arg14 : index
          %68 = memref.load %arg7[%67] : memref<?xf64>
          %69 = arith.mulf %arg3, %arg4 : f64
          %70 = arith.mulf %69, %cst : f64
          %71 = arith.addf %55, %68 : f64
          %72 = arith.addf %59, %64 : f64
          %73 = arith.subf %71, %72 : f64
          %74 = arith.divf %73, %70 : f64
          %75 = arith.addi %arg14, %c1 : index
          %76 = arith.addi %37, %75 : index
          %77 = memref.load %arg7[%76] : memref<?xf64>
          %78 = arith.addi %43, %75 : index
          %79 = memref.load %arg7[%78] : memref<?xf64>
          %80 = arith.addi %arg14, %c-1 : index
          %81 = arith.addi %37, %80 : index
          %82 = memref.load %arg7[%81] : memref<?xf64>
          %83 = arith.addi %43, %80 : index
          %84 = memref.load %arg7[%83] : memref<?xf64>
          %85 = arith.mulf %arg3, %arg5 : f64
          %86 = arith.mulf %85, %cst : f64
          %87 = arith.addf %77, %84 : f64
          %88 = arith.addf %79, %82 : f64
          %89 = arith.subf %87, %88 : f64
          %90 = arith.divf %89, %86 : f64
          %91 = arith.addi %5, %51 : index
          %92 = arith.muli %91, %arg2 : index
          %93 = arith.addi %92, %arg14 : index
          %94 = memref.load %arg7[%93] : memref<?xf64>
          %95 = arith.addi %5, %60 : index
          %96 = arith.muli %95, %arg2 : index
          %97 = arith.addi %96, %arg14 : index
          %98 = memref.load %arg7[%97] : memref<?xf64>
          %99 = arith.mulf %arg4, %arg4 : f64
          %100 = arith.addf %94, %98 : f64
          %101 = arith.subf %100, %47 : f64
          %102 = arith.divf %101, %99 : f64
          %103 = arith.addi %92, %75 : index
          %104 = memref.load %arg7[%103] : memref<?xf64>
          %105 = arith.addi %96, %75 : index
          %106 = memref.load %arg7[%105] : memref<?xf64>
          %107 = arith.addi %92, %80 : index
          %108 = memref.load %arg7[%107] : memref<?xf64>
          %109 = arith.addi %96, %80 : index
          %110 = memref.load %arg7[%109] : memref<?xf64>
          %111 = arith.mulf %arg4, %arg5 : f64
          %112 = arith.mulf %111, %cst : f64
          %113 = arith.addf %104, %110 : f64
          %114 = arith.addf %106, %108 : f64
          %115 = arith.subf %113, %114 : f64
          %116 = arith.divf %115, %112 : f64
          %117 = arith.addi %7, %75 : index
          %118 = memref.load %arg7[%117] : memref<?xf64>
          %119 = arith.addi %7, %80 : index
          %120 = memref.load %arg7[%119] : memref<?xf64>
          %121 = arith.mulf %arg5, %arg5 : f64
          %122 = arith.addf %118, %120 : f64
          %123 = arith.subf %122, %47 : f64
          %124 = arith.divf %123, %121 : f64
          %125 = arith.mulf %9, %50 : f64
          %126 = arith.mulf %11, %74 : f64
          %127 = arith.mulf %14, %90 : f64
          %128 = arith.mulf %17, %74 : f64
          %129 = arith.mulf %20, %102 : f64
          %130 = arith.mulf %23, %116 : f64
          %131 = arith.mulf %26, %90 : f64
          %132 = arith.mulf %29, %116 : f64
          %133 = arith.mulf %32, %124 : f64
          %134 = arith.addf %125, %126 : f64
          %135 = arith.addf %134, %127 : f64
          %136 = arith.addf %128, %129 : f64
          %137 = arith.addf %136, %130 : f64
          %138 = arith.addf %135, %137 : f64
          %139 = arith.addf %131, %132 : f64
          %140 = arith.addf %139, %133 : f64
          %141 = arith.addf %138, %140 : f64
          %142 = memref.load %arg6[%8] : memref<?xf64>
          %143 = memref.load %arg9[%8] : memref<?xf64>
          %144 = memref.load %arg9[%10] : memref<?xf64>
          %145 = memref.load %arg9[%13] : memref<?xf64>
          %146 = memref.load %arg9[%16] : memref<?xf64>
          %147 = memref.load %arg9[%19] : memref<?xf64>
          %148 = memref.load %arg9[%22] : memref<?xf64>
          %149 = memref.load %arg9[%25] : memref<?xf64>
          %150 = memref.load %arg9[%28] : memref<?xf64>
          %151 = memref.load %arg9[%31] : memref<?xf64>
          %152 = arith.mulf %142, %143 : f64
          %153 = arith.mulf %142, %144 : f64
          %154 = arith.mulf %142, %145 : f64
          %155 = arith.mulf %142, %146 : f64
          %156 = arith.mulf %142, %147 : f64
          %157 = arith.mulf %142, %148 : f64
          %158 = arith.mulf %142, %149 : f64
          %159 = arith.mulf %142, %150 : f64
          %160 = arith.mulf %142, %151 : f64
          %161 = arith.addf %50, %152 : f64
          %162 = arith.addf %74, %155 : f64
          %163 = arith.addf %90, %158 : f64
          %164 = arith.addf %74, %153 : f64
          %165 = arith.addf %102, %156 : f64
          %166 = arith.addf %116, %159 : f64
          %167 = arith.addf %90, %154 : f64
          %168 = arith.addf %116, %157 : f64
          %169 = arith.addf %124, %160 : f64
          memref.store %141, %arg8[%8] : memref<?xf64>
          memref.store %161, %arg11[%8] : memref<?xf64>
          memref.store %164, %arg11[%10] : memref<?xf64>
          memref.store %167, %arg11[%13] : memref<?xf64>
          memref.store %162, %arg11[%16] : memref<?xf64>
          memref.store %165, %arg11[%19] : memref<?xf64>
          memref.store %168, %arg11[%22] : memref<?xf64>
          memref.store %163, %arg11[%25] : memref<?xf64>
          memref.store %166, %arg11[%28] : memref<?xf64>
          memref.store %169, %arg11[%31] : memref<?xf64>
        }
      }
    }
    return
  }
}

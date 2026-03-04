// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

// ===--- Named Linalg Ops ---=============================================//

// CHECK-LABEL: func.func @test_add
func.func @test_add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.add
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_add_broadcast
func.func @test_add_broadcast(%arg0: tensor<64x128xf32>, %arg1: tensor<1x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.broadcast
  // CHECK: linalg.add
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<1x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_subtract
func.func @test_subtract(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.sub
  %0 = "ttir.subtract"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_multiply
func.func @test_multiply(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.mul
  %0 = "ttir.multiply"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_div
func.func @test_div(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.div
  %0 = "ttir.div"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_pow
func.func @test_pow(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.powf
  %0 = "ttir.pow"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// ===--- Linalg Generic + Math Ops ---====================================//

// CHECK-LABEL: func.func @test_atan2
func.func @test_atan2(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.generic
  // CHECK: math.atan2
  %0 = "ttir.atan2"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// ===--- Linalg Generic + Arith Ops ---===================================//

// CHECK-LABEL: func.func @test_remainder_f32
func.func @test_remainder_f32(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.remf
  %0 = "ttir.remainder"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_remainder_i32
func.func @test_remainder_i32(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
  // CHECK: linalg.generic
  // CHECK: arith.remsi
  %0 = "ttir.remainder"(%arg0, %arg1) : (tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
  return %0 : tensor<64x128xi32>
}

// CHECK-LABEL: func.func @test_bitwise_and
func.func @test_bitwise_and(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
  // CHECK: linalg.generic
  // CHECK: arith.andi
  %0 = "ttir.bitwise_and"(%arg0, %arg1) : (tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
  return %0 : tensor<64x128xi32>
}

// CHECK-LABEL: func.func @test_bitwise_or
func.func @test_bitwise_or(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
  // CHECK: linalg.generic
  // CHECK: arith.ori
  %0 = "ttir.bitwise_or"(%arg0, %arg1) : (tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
  return %0 : tensor<64x128xi32>
}

// CHECK-LABEL: func.func @test_bitwise_xor
func.func @test_bitwise_xor(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
  // CHECK: linalg.generic
  // CHECK: arith.xori
  %0 = "ttir.bitwise_xor"(%arg0, %arg1) : (tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
  return %0 : tensor<64x128xi32>
}

// CHECK-LABEL: func.func @test_logical_left_shift
func.func @test_logical_left_shift(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
  // CHECK: linalg.generic
  // CHECK: arith.shli
  %0 = "ttir.logical_left_shift"(%arg0, %arg1) : (tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
  return %0 : tensor<64x128xi32>
}

// CHECK-LABEL: func.func @test_logical_right_shift
func.func @test_logical_right_shift(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
  // CHECK: linalg.generic
  // CHECK: arith.shrui
  %0 = "ttir.logical_right_shift"(%arg0, %arg1) : (tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
  return %0 : tensor<64x128xi32>
}

// ===--- TOSA 1:1 Binary Ops ---==========================================//

// CHECK-LABEL: func.func @test_maximum
func.func @test_maximum(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.maximum
  %0 = "ttir.maximum"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_minimum
func.func @test_minimum(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.minimum
  %0 = "ttir.minimum"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// ===--- Comparison Ops ---================================================//

// CHECK-LABEL: func.func @test_eq
func.func @test_eq(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.equal
  // CHECK: tosa.cast
  %0 = "ttir.eq"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_ne
func.func @test_ne(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.equal
  // CHECK: tosa.logical_not
  // CHECK: tosa.cast
  %0 = "ttir.ne"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_gt
func.func @test_gt(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.greater
  // CHECK: tosa.cast
  %0 = "ttir.gt"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_ge
func.func @test_ge(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.greater_equal
  // CHECK: tosa.cast
  %0 = "ttir.ge"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_lt
func.func @test_lt(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.greater
  // CHECK: tosa.cast
  %0 = "ttir.lt"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_le
func.func @test_le(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.greater_equal
  // CHECK: tosa.cast
  %0 = "ttir.le"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// ===--- Logical Binary Ops ---============================================//

// CHECK-LABEL: func.func @test_logical_and
func.func @test_logical_and(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.equal
  // CHECK: tosa.logical_not
  // CHECK: tosa.equal
  // CHECK: tosa.logical_not
  // CHECK: tosa.logical_and
  // CHECK: tosa.cast
  %0 = "ttir.logical_and"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_logical_or
func.func @test_logical_or(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.equal
  // CHECK: tosa.logical_not
  // CHECK: tosa.equal
  // CHECK: tosa.logical_not
  // CHECK: tosa.logical_or
  // CHECK: tosa.cast
  %0 = "ttir.logical_or"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_logical_xor
func.func @test_logical_xor(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.equal
  // CHECK: tosa.logical_not
  // CHECK: tosa.equal
  // CHECK: tosa.logical_not
  // CHECK: tosa.logical_xor
  // CHECK: tosa.cast
  %0 = "ttir.logical_xor"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// ===--- Compound Binary Ops ---==========================================//

// CHECK-LABEL: func.func @test_gelu_bw
func.func @test_gelu_bw(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.mul
  // CHECK: tosa.erf
  // CHECK: tosa.add
  // CHECK: tosa.mul
  // CHECK: tosa.mul
  // CHECK: tosa.mul
  // CHECK: tosa.exp
  // CHECK: tosa.mul
  // CHECK: tosa.mul
  // CHECK: tosa.add
  // CHECK: tosa.mul
  %0 = "ttir.gelu_bw"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @test_gelu_bw_tanh
func.func @test_gelu_bw_tanh(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: tosa.mul
  // CHECK: tosa.mul
  // CHECK: tosa.add
  // CHECK: tosa.mul
  // CHECK: tosa.tanh
  // CHECK: tosa.mul
  // CHECK: tosa.negate
  // CHECK: tosa.add
  // CHECK: tosa.add
  // CHECK: tosa.mul
  // CHECK: tosa.mul
  // CHECK: tosa.add
  // CHECK: tosa.mul
  // CHECK: tosa.mul
  // CHECK: tosa.mul
  // CHECK: tosa.add
  // CHECK: tosa.mul
  %0 = "ttir.gelu_bw"(%arg0, %arg1) <{approximate = "tanh"}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

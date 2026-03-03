// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOLINALG_UTILS_H
#define TTMLIR_CONVERSION_TTIRTOLINALG_UTILS_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tt::ttir_to_linalg {

//===----------------------------------------------------------------------===//
// Broadcasting helpers
//===----------------------------------------------------------------------===//

// Get the dimensions to broadcast.
//
// This function calculates the dimensions to broadcast. We assume that input
// and target shapes are broadcastable. For example if input shape is [4, 1, 3]
// and we want to broadcast to [1, 4, 5, 3], the function will return [0, 2]
// since we want to broadcast 0th and 2nd dimension of result shape.
inline SmallVector<int64_t, 2> getBroadcastDims(ArrayRef<int64_t> inputShape,
                                                ArrayRef<int64_t> targetShape) {
  const int64_t sizeDiff = targetShape.size() - inputShape.size();
  assert(sizeDiff >= 0 && "targetShape cannot be smaller than inputShape!");

  // Create padded input shape by prepending 1s.
  SmallVector<int64_t> paddedInput;
  paddedInput.append(sizeDiff, 1); // Prepend with 1s
  paddedInput.append(inputShape.begin(), inputShape.end());

  // Find broadcast dimensions we want to broadcast along (including padding
  // dimensions).
  SmallVector<int64_t, 2> broadcastDims;
  for (const auto &it : llvm::enumerate(llvm::zip(paddedInput, targetShape))) {
    const size_t i = it.index();
    const auto &[inputDim, targetDim] = it.value();
    // Prepended dimensions are always broadcasted.
    if (i < static_cast<size_t>(sizeDiff) || inputDim != targetDim) {
      broadcastDims.push_back(i);
    }
  }

  return broadcastDims;
}

// Get the dimensions to collapse.
//
// This function calculates the dimensions to collapse. We assume that input
// and target shapes are broadcastable. linalg.broadcast requires that input
// tensor only contains dimensions that won't be broadcasted in input tensor.
// For example if input shape is [4, 1, 3] and we want to broadcast to [4, 5,
// 3], then we need to collapse the first dimension of input tensor to [4, 3].
// This function calculates the dimensions to collapse. In case above, we will
// return [[0], [1, 2]].
inline SmallVector<SmallVector<int64_t, 2>, 2>
getCollapseDims(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> targetShape) {
  // Calculate the size difference.
  const size_t sizeDiff = targetShape.size() - inputShape.size();

  // Create the padded input shape by prepending 1s.
  SmallVector<int64_t> paddedInput(sizeDiff, 1);
  paddedInput.append(inputShape.begin(), inputShape.end());

  SmallVector<int64_t, 2> collapseDims;
  SmallVector<SmallVector<int64_t, 2>, 2> reassocIndexes;
  for (size_t i = sizeDiff; i < targetShape.size(); ++i) {
    const size_t inputDim = paddedInput[i];
    const size_t targetDim = targetShape[i];
    // Adjust the index to account for the prepended dimensions
    // that are not part of the input shape.
    collapseDims.push_back(i - sizeDiff);
    if (inputDim == targetDim) {
      reassocIndexes.push_back(collapseDims);
      collapseDims.clear();
    }
  }

  if (!collapseDims.empty()) {
    if (reassocIndexes.empty()) {
      reassocIndexes.push_back(collapseDims);
    } else {
      reassocIndexes.back().append(collapseDims.begin(), collapseDims.end());
    }
  }

  return reassocIndexes;
}

// Broadcast input to targetShape using linalg.broadcast if needed.
// Returns the input unchanged if shapes already match.
Value broadcastToShape(Value input, ArrayRef<int64_t> targetShape, Location loc,
                       ConversionPatternRewriter &rewriter);

//===----------------------------------------------------------------------===//
// Tensor/TOSA helpers
//===----------------------------------------------------------------------===//

// Convert a tensor of floating-point values to a tensor of boolean values
// by comparing with zero; zero is false and nonzero is true.
Value convertToBooleanTensor(Value input, Location loc,
                             ConversionPatternRewriter &rewriter);

// Helper function to create DenseElementsAttr with a specific value based on
// element type.
inline DenseElementsAttr createDenseElementsAttr(RankedTensorType resultType,
                                                 double value) {
  auto elementType = resultType.getElementType();
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    return SplatElementsAttr::get(resultType, FloatAttr::get(floatType, value));
  }
  if (isa<IntegerType>(elementType)) {
    return SplatElementsAttr::get(
        resultType, IntegerAttr::get(elementType, static_cast<int64_t>(value)));
  }
  return {};
}

// Helper to create a ranked TOSA constant with shape [1, 1, ...] matching rank.
inline Value createTosaFloatConst(ConversionPatternRewriter &rewriter,
                                  Location loc, Type elementType, int64_t rank,
                                  double value) {
  SmallVector<int64_t> shape(rank, 1);
  auto type = RankedTensorType::get(shape, elementType);
  auto attr =
      DenseElementsAttr::get(type, rewriter.getFloatAttr(elementType, value));
  return rewriter.create<tosa::ConstOp>(loc, type, attr);
}

// Helper to create the TOSA mul shift operand (i8 zero tensor).
inline Value createTosaMulShift(ConversionPatternRewriter &rewriter,
                                Location loc) {
  auto type = RankedTensorType::get({1}, rewriter.getI8Type());
  auto attr = DenseElementsAttr::get(type, rewriter.getI8IntegerAttr(0));
  return rewriter.create<tosa::ConstOp>(loc, type, attr);
}

} // namespace mlir::tt::ttir_to_linalg

#endif // TTMLIR_CONVERSION_TTIRTOLINALG_UTILS_H

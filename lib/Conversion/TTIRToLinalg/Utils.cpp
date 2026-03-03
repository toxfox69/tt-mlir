// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinalg/Utils.h"

namespace mlir::tt::ttir_to_linalg {

Value broadcastToShape(Value input, ArrayRef<int64_t> targetShape, Location loc,
                       ConversionPatternRewriter &rewriter) {
  auto inputType = cast<RankedTensorType>(input.getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();

  SmallVector<int64_t, 2> broadcastDims =
      getBroadcastDims(inputShape, targetShape);

  // No broadcasting needed.
  if (broadcastDims.empty()) {
    return input;
  }

  // If we need to broadcast along all dims, then we need to collapse to a
  // scalar via empty collapseDimGroups.
  SmallVector<SmallVector<int64_t, 2>, 2> collapseDimGroups =
      (broadcastDims.size() != targetShape.size())
          ? getCollapseDims(inputShape, targetShape)
          : SmallVector<SmallVector<int64_t, 2>, 2>();

  Value broadcastInput = input;
  // The broadcast op requires we actually collapse any dimensions with
  // size 1 we want to broadcast along.
  if (collapseDimGroups.size() != inputShape.size()) {
    broadcastInput =
        rewriter.create<tensor::CollapseShapeOp>(loc, input, collapseDimGroups);
  }

  auto initTensor = rewriter.create<ttir::EmptyOp>(loc, targetShape,
                                                   inputType.getElementType());
  auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
      loc, broadcastInput, initTensor.getResult(), broadcastDims);

  return broadcastOp.getResults().front();
}

Value convertToBooleanTensor(Value input, Location loc,
                             ConversionPatternRewriter &rewriter) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    return input;
  }

  // If it's already a boolean tensor, return it as is
  if (inputType.getElementType().isInteger(1)) {
    return input;
  }

  auto elementType = inputType.getElementType();
  assert(elementType.isF32() && "Only f32 element type is supported");

  // Create zero constant.
  SmallVector<int64_t> zeroShape(inputType.getRank(), 1);
  auto zeroType = RankedTensorType::get(zeroShape, elementType);
  DenseElementsAttr zeroAttr =
      DenseElementsAttr::get(zeroType, rewriter.getF32FloatAttr(0.0f));
  auto zeroConst = rewriter.create<tosa::ConstOp>(loc, zeroType, zeroAttr);

  // For logical operations, non-zero means true.
  // So we need: (input != 0) which we get by computing !(input == 0).
  auto boolType =
      RankedTensorType::get(inputType.getShape(), rewriter.getIntegerType(1));
  auto equalZero =
      rewriter.create<tosa::EqualOp>(loc, boolType, input, zeroConst);
  // Then use LogicalNotOp to invert it, giving us (input != 0).
  auto notEqualZero =
      rewriter.create<tosa::LogicalNotOp>(loc, boolType, equalZero);

  return notEqualZero;
}

} // namespace mlir::tt::ttir_to_linalg

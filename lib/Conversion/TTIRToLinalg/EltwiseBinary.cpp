// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToLinalg/EltwiseBinary.h"
#include "ttmlir/Conversion/TTIRToLinalg/Utils.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cmath>
#include <cstdint>

namespace mlir::tt::ttir_to_linalg {

// Conversion patterns for TTIR binary elementwise ops are organized by
// implementation strategy, in order of preference:
//
// 1. TOSA 1:1        — Direct mapping to a single TOSA op.
//                      Preferred when a TOSA equivalent exists.
// 2. Named linalg    — Direct mapping to a named linalg op (e.g. linalg.add).
//                      Used when no TOSA equivalent exists but a named linalg
//                      op does. Supports implicit broadcasting.
// 3. linalg.generic + math/arith — A linalg.generic body containing a single
//                      math or arith dialect op. Used for ops with no TOSA or
//                      named linalg equivalent.
// 4. Custom          — Multi-op sequences in TOSA, linalg, or arith dialects.
//                      Used for compound operations (e.g. gelu_bw,
//                      clamp_tensor).

//===----------------------------------------------------------------------===//
// TOSA Binary Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
template <typename TTIROpTy, typename TosaOpTy>
class TosaElementwiseBinaryOpConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    auto result = rewriter.create<TosaOpTy>(op.getLoc(), resultType,
                                            ValueRange{lhs, rhs});

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

// Direct comparison operations (where TTIR and TOSA ops match directly).
namespace {
template <typename TTIROpTy, typename TosaOpTy>
class DirectComparisonOpConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Create the TOSA comparison operation
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));
    auto boolResult =
        rewriter.create<TosaOpTy>(op.getLoc(), boolType, lhs, rhs);

    // Convert boolean result to original type using cast.
    auto result =
        rewriter.create<tosa::CastOp>(op.getLoc(), resultType, boolResult);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Swapped comparison operations (where TTIR and TOSA ops have swapped operands
// e.g. ttir.lt must use inverted tosa.greater).
template <typename TTIROpTy, typename TosaOpTy>
class SwappedComparisonOpConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Create the TOSA comparison operation with swapped operands
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));
    auto boolResult =
        rewriter.create<TosaOpTy>(op.getLoc(), boolType, rhs, lhs);

    // Convert boolean result to original type using cast.
    auto result =
        rewriter.create<tosa::CastOp>(op.getLoc(), resultType, boolResult);

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Negated comparison operations (where TTIR op is the negation of a TOSA op,
// e.g. ttir.not_equal).
template <typename TTIROpTy, typename TosaOpTy>
class NegatedComparisonOpConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Create the TOSA comparison operation
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));
    auto boolResult =
        rewriter.create<TosaOpTy>(op.getLoc(), boolType, lhs, rhs);

    // Negate the boolean result
    auto notResult =
        rewriter.create<tosa::LogicalNotOp>(op.getLoc(), boolType, boolResult);

    // Convert boolean result to original type using cast.
    auto result =
        rewriter.create<tosa::CastOp>(op.getLoc(), resultType, notResult);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

// Logical binary operations pattern (LogicalAnd, LogicalOr, LogicalXor).
// These operations:
// 1. Convert float inputs to boolean (non-zero = true)
// 2. Apply the TOSA logical operation
// 3. Convert boolean result back to float (true = 1.0, false = 0.0)
namespace {
template <typename TTIROpTy, typename TosaOpTy>
class LogicalBinaryOpConversionPattern : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    assert(resultType && "Result type must be a ranked tensor type.");

    // Convert both inputs to boolean tensors.
    Value boolLhs = convertToBooleanTensor(lhs, op.getLoc(), rewriter);
    Value boolRhs = convertToBooleanTensor(rhs, op.getLoc(), rewriter);

    // Get the boolean type for the intermediate result.
    auto boolType = RankedTensorType::get(resultType.getShape(),
                                          rewriter.getIntegerType(1));

    // Apply the logical operation to the boolean tensors.
    auto logicalResult =
        rewriter.create<TosaOpTy>(op.getLoc(), boolType, boolLhs, boolRhs);

    // Convert boolean result back to original type using cast.
    auto result =
        rewriter.create<tosa::CastOp>(op.getLoc(), resultType, logicalResult);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Named Linalg Op Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
// General elementwise conversion pattern for binary ops lowered to named linalg
// ops. Supports implicit broadcasting by broadcasting both inputs to the result
// shape.
template <typename TTIROpTy, typename LinalgOpTy,
          typename OpAdaptor = typename TTIROpTy::Adaptor>
class ElementwiseBinaryOpConversionPattern
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    RankedTensorType lhsType =
        cast<RankedTensorType>(adaptor.getLhs().getType());
    RankedTensorType rhsType =
        cast<RankedTensorType>(adaptor.getRhs().getType());

    // First, compute broadcasted shape from operands.

    ArrayRef<int64_t> lhsShape = lhsType.getShape();
    ArrayRef<int64_t> rhsShape = rhsType.getShape();

    SmallVector<int64_t> broadcastedShape;
    if (!OpTrait::util::getBroadcastedShape(lhsShape, rhsShape,
                                            broadcastedShape)) {
      return rewriter.notifyMatchFailure(op, "Operands are not broadcastable!");
    }

    // Rewrite inputs to target dims with broadcast and collapse shape ops, as
    // needed.
    SmallVector<Value, 2> inputs{adaptor.getLhs(), adaptor.getRhs()};
    SmallVector<Value, 2> broadcastedInputs;
    for (Value input : inputs) {
      auto inputRankedTensorType = dyn_cast<RankedTensorType>(input.getType());
      assert(inputRankedTensorType &&
             "Binary element-wise operations must be ranked tensor types!");

      // Insert and use a broadcast op if input does not perfectly match target
      // shape.
      SmallVector<int64_t, 2> broadcastDims =
          getBroadcastDims(inputRankedTensorType.getShape(), broadcastedShape);

      // If we need to broadcast along all dims, then we need to collapse to a
      // scalar via empty collapseDimGroups.
      SmallVector<SmallVector<int64_t, 2>, 2> collapseDimGroups =
          (broadcastDims.size() != broadcastedShape.size())
              ? getCollapseDims(inputRankedTensorType.getShape(),
                                broadcastedShape)
              : SmallVector<SmallVector<int64_t, 2>, 2>();

      if (!broadcastDims.empty()) {
        Value broadcastInput = input;
        // The broadcast op requires we actually collapse any dimensions with
        // size 1 we want to broadcast along.
        if (collapseDimGroups.size() !=
            inputRankedTensorType.getShape().size()) {
          broadcastInput = rewriter.create<tensor::CollapseShapeOp>(
              loc, input, collapseDimGroups);
        }
        auto initTensor = rewriter.create<ttir::EmptyOp>(
            loc, broadcastedShape, inputRankedTensorType.getElementType());
        auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
            loc, broadcastInput, initTensor.getResult(), broadcastDims);
        broadcastedInputs.push_back(broadcastOp.getResults().front());
      } else {
        broadcastedInputs.push_back(input);
      }
    }

    // Perform the actual op substitution, using broadcasted operands when
    // needed.
    auto resultType = mlir::cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));

    auto output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());
    rewriter.replaceOpWithNewOp<LinalgOpTy>(op, resultType, broadcastedInputs,
                                            output.getResult());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Linalg Generic + Math/Arith Dialect Patterns
//===----------------------------------------------------------------------===//

// Base class for TTIR binary ops lowered via linalg.generic. Subclasses only
// need to implement buildBody() to emit the scalar computation. Supports
// implicit broadcasting by broadcasting both inputs to result shape.
namespace {
template <typename TTIROpTy>
class ElementwiseBinaryLinalgGenericBase
    : public OpConversionPattern<TTIROpTy> {
public:
  using OpConversionPattern<TTIROpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TTIROpTy op, typename TTIROpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }

    Location loc = op.getLoc();
    Value lhs = broadcastToShape(adaptor.getLhs(), resultType.getShape(), loc,
                                 rewriter);
    Value rhs = broadcastToShape(adaptor.getRhs(), resultType.getShape(), loc,
                                 rewriter);

    int64_t rank = resultType.getRank();
    auto indexingMap =
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{lhs, rhs}, ValueRange{emptyTensor},
        SmallVector<AffineMap>{indexingMap, indexingMap, indexingMap},
        iteratorTypes, [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value result = buildBody(b, nestedLoc, args, resultType);
          b.create<linalg::YieldOp>(nestedLoc, result);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }

protected:
  virtual Value buildBody(OpBuilder &b, Location loc, ValueRange args,
                          RankedTensorType resultType) const = 0;
};
} // namespace

// Template for TTIR ops that map 1:1 to a math dialect op via linalg.generic.
namespace {
template <typename TTIROpTy, typename MathOpTy>
class ElementwiseBinaryMathOpConversionPattern
    : public ElementwiseBinaryLinalgGenericBase<TTIROpTy> {
public:
  using ElementwiseBinaryLinalgGenericBase<
      TTIROpTy>::ElementwiseBinaryLinalgGenericBase;

protected:
  Value buildBody(OpBuilder &b, Location loc, ValueRange args,
                  RankedTensorType /*resultType*/) const override {
    return b.create<MathOpTy>(loc, args[0], args[1]);
  }
};
} // namespace

// Template for TTIR ops that map 1:1 to an arith dialect op via linalg.generic.
namespace {
template <typename TTIROpTy, typename ArithOpTy>
class ElementwiseBinaryArithOpConversionPattern
    : public ElementwiseBinaryLinalgGenericBase<TTIROpTy> {
public:
  using ElementwiseBinaryLinalgGenericBase<
      TTIROpTy>::ElementwiseBinaryLinalgGenericBase;

protected:
  Value buildBody(OpBuilder &b, Location loc, ValueRange args,
                  RankedTensorType /*resultType*/) const override {
    return b.create<ArithOpTy>(loc, args[0], args[1]);
  }
};
} // namespace

// Remainder: arith.remf for floats, arith.remsi for signed ints.
namespace {
class RemainderOpConversionPattern
    : public ElementwiseBinaryLinalgGenericBase<ttir::RemainderOp> {
public:
  using ElementwiseBinaryLinalgGenericBase::ElementwiseBinaryLinalgGenericBase;

protected:
  Value buildBody(OpBuilder &b, Location loc, ValueRange args,
                  RankedTensorType resultType) const override {
    if (isa<FloatType>(resultType.getElementType())) {
      return b.create<arith::RemFOp>(loc, args[0], args[1]);
    }
    return b.create<arith::RemSIOp>(loc, args[0], args[1]);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Compound Binary Patterns (multi-op TOSA sequences)
//===----------------------------------------------------------------------===//

// GeluBackward: gelu_bw(grad, x) for both exact and tanh approximations.
namespace {
class GeluBackwardOpConversionPattern
    : public OpConversionPattern<ttir::GeluBackwardOp> {
public:
  using OpConversionPattern<ttir::GeluBackwardOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::GeluBackwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value grad = adaptor.getLhs();
    Value x = adaptor.getRhs();
    auto resultType = dyn_cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type must be a ranked tensor type");
    }

    Location loc = op.getLoc();

    // Broadcast both inputs to the result shape for implicit broadcasting.
    grad = broadcastToShape(grad, resultType.getShape(), loc, rewriter);
    x = broadcastToShape(x, resultType.getShape(), loc, rewriter);

    auto approximate = op.getApproximate();

    if (approximate == "tanh") {
      return matchAndRewriteTanh(grad, x, resultType, loc, op, rewriter);
    }
    return matchAndRewriteExact(grad, x, resultType, loc, op, rewriter);
  }

private:
  // gelu_bw(grad, x) = grad * (cdf(x) + x * pdf(x))
  // where cdf(x) = 0.5 * (1 + erf(x / sqrt(2)))
  //       pdf(x) = exp(-x^2/2) / sqrt(2*pi)
  LogicalResult
  matchAndRewriteExact(Value grad, Value x, RankedTensorType resultType,
                       Location loc, ttir::GeluBackwardOp op,
                       ConversionPatternRewriter &rewriter) const {
    auto elemTy = resultType.getElementType();
    int64_t rank = resultType.getRank();
    Value shift = createTosaMulShift(rewriter, loc);

    Value half = createTosaFloatConst(rewriter, loc, elemTy, rank, 0.5);
    Value one = createTosaFloatConst(rewriter, loc, elemTy, rank, 1.0);
    Value invSqrt2 =
        createTosaFloatConst(rewriter, loc, elemTy, rank, M_SQRT1_2);
    Value negHalf = createTosaFloatConst(rewriter, loc, elemTy, rank, -0.5);
    Value invSqrt2Pi = createTosaFloatConst(rewriter, loc, elemTy, rank,
                                            1.0 / std::sqrt(2.0 * M_PI));

    // cdf = 0.5 * (1 + erf(x * invSqrt2))
    Value xScaled =
        rewriter.create<tosa::MulOp>(loc, resultType, x, invSqrt2, shift);
    Value erfVal = rewriter.create<tosa::ErfOp>(loc, resultType, xScaled);
    Value onePlusErf =
        rewriter.create<tosa::AddOp>(loc, resultType, one, erfVal);
    Value cdf =
        rewriter.create<tosa::MulOp>(loc, resultType, half, onePlusErf, shift);

    // pdf = exp(-x^2/2) / sqrt(2*pi)
    Value xSq = rewriter.create<tosa::MulOp>(loc, resultType, x, x, shift);
    Value negHalfXSq =
        rewriter.create<tosa::MulOp>(loc, resultType, negHalf, xSq, shift);
    Value expVal = rewriter.create<tosa::ExpOp>(loc, resultType, negHalfXSq);
    Value pdf = rewriter.create<tosa::MulOp>(loc, resultType, invSqrt2Pi,
                                             expVal, shift);

    // result = grad * (cdf + x * pdf)
    Value xTimesPdf =
        rewriter.create<tosa::MulOp>(loc, resultType, x, pdf, shift);
    Value cdfPlusXPdf =
        rewriter.create<tosa::AddOp>(loc, resultType, cdf, xTimesPdf);
    rewriter.replaceOpWithNewOp<tosa::MulOp>(op, resultType, grad, cdfPlusXPdf,
                                             shift);
    return success();
  }

  // gelu_bw with tanh approximation:
  // k = sqrt(2/pi), a = 0.044715
  // inner = k * (x + a * x^3)
  // tanh_val = tanh(inner)
  // left = 0.5 * (1 + tanh_val)
  // right = 0.5 * x * (1 - tanh_val^2) * k * (1 + 3*a*x^2)
  // gelu_bw = grad * (left + right)
  LogicalResult matchAndRewriteTanh(Value grad, Value x,
                                    RankedTensorType resultType, Location loc,
                                    ttir::GeluBackwardOp op,
                                    ConversionPatternRewriter &rewriter) const {
    auto elemTy = resultType.getElementType();
    int64_t rank = resultType.getRank();
    Value shift = createTosaMulShift(rewriter, loc);

    Value k = createTosaFloatConst(rewriter, loc, elemTy, rank,
                                   std::sqrt(2.0 / M_PI));
    Value a = createTosaFloatConst(rewriter, loc, elemTy, rank, 0.044715);
    Value threeA =
        createTosaFloatConst(rewriter, loc, elemTy, rank, 3.0 * 0.044715);
    Value half = createTosaFloatConst(rewriter, loc, elemTy, rank, 0.5);
    Value one = createTosaFloatConst(rewriter, loc, elemTy, rank, 1.0);

    // x^2
    Value xSq = rewriter.create<tosa::MulOp>(loc, resultType, x, x, shift);
    // a * x^2 * x = a * x^3
    Value aXSq = rewriter.create<tosa::MulOp>(loc, resultType, a, xSq, shift);
    Value aXCub = rewriter.create<tosa::MulOp>(loc, resultType, aXSq, x, shift);
    // inner = k * (x + a*x^3)
    Value xPlusAXCub = rewriter.create<tosa::AddOp>(loc, resultType, x, aXCub);
    Value inner =
        rewriter.create<tosa::MulOp>(loc, resultType, k, xPlusAXCub, shift);

    // tanh_val = tanh(inner)
    Value tanhVal = rewriter.create<tosa::TanhOp>(loc, resultType, inner);

    // sech^2 = 1 - tanh^2
    Value tanhSq =
        rewriter.create<tosa::MulOp>(loc, resultType, tanhVal, tanhVal, shift);
    Value negTanhSq = rewriter.create<tosa::NegateOp>(loc, resultType, tanhSq);
    Value sechSq =
        rewriter.create<tosa::AddOp>(loc, resultType, one, negTanhSq);

    // left = 0.5 * (1 + tanh_val)
    Value onePlusTanh =
        rewriter.create<tosa::AddOp>(loc, resultType, one, tanhVal);
    Value left =
        rewriter.create<tosa::MulOp>(loc, resultType, half, onePlusTanh, shift);

    // right = 0.5 * x * sech^2 * k * (1 + 3*a*x^2)
    Value threeAXSq =
        rewriter.create<tosa::MulOp>(loc, resultType, threeA, xSq, shift);
    Value onePlus3AXSq =
        rewriter.create<tosa::AddOp>(loc, resultType, one, threeAXSq);
    Value sechK =
        rewriter.create<tosa::MulOp>(loc, resultType, sechSq, k, shift);
    Value sechKTerm = rewriter.create<tosa::MulOp>(loc, resultType, sechK,
                                                   onePlus3AXSq, shift);
    Value xTerm =
        rewriter.create<tosa::MulOp>(loc, resultType, x, sechKTerm, shift);
    Value right =
        rewriter.create<tosa::MulOp>(loc, resultType, half, xTerm, shift);

    // gelu_bw = grad * (left + right)
    Value leftPlusRight =
        rewriter.create<tosa::AddOp>(loc, resultType, left, right);
    rewriter.replaceOpWithNewOp<tosa::MulOp>(op, resultType, grad,
                                             leftPlusRight, shift);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void populateTTIRToLinalgEltwiseBinaryPatterns(MLIRContext *ctx,
                                               RewritePatternSet &patterns,
                                               TypeConverter &typeConverter) {
  // Named linalg ops (with implicit broadcasting support)
  patterns.add<
      ElementwiseBinaryOpConversionPattern<ttir::AddOp, linalg::AddOp>,
      ElementwiseBinaryOpConversionPattern<ttir::SubtractOp, linalg::SubOp>,
      ElementwiseBinaryOpConversionPattern<ttir::MultiplyOp, linalg::MulOp>,
      ElementwiseBinaryOpConversionPattern<ttir::DivOp, linalg::DivOp>,
      ElementwiseBinaryOpConversionPattern<ttir::PowOp, linalg::PowFOp>>(
      typeConverter, ctx);

  // linalg.generic + math dialect ops
  patterns.add<
      ElementwiseBinaryMathOpConversionPattern<ttir::Atan2Op, math::Atan2Op>>(
      typeConverter, ctx);

  // linalg.generic + arith dialect ops (integer-only bitwise and shift ops)
  patterns.add<ElementwiseBinaryArithOpConversionPattern<ttir::BitwiseAndOp,
                                                         arith::AndIOp>,
               ElementwiseBinaryArithOpConversionPattern<ttir::BitwiseOrOp,
                                                         arith::OrIOp>,
               ElementwiseBinaryArithOpConversionPattern<ttir::BitwiseXorOp,
                                                         arith::XOrIOp>,
               ElementwiseBinaryArithOpConversionPattern<
                   ttir::LogicalLeftShiftOp, arith::ShLIOp>,
               ElementwiseBinaryArithOpConversionPattern<
                   ttir::LogicalRightShiftOp, arith::ShRUIOp>>(typeConverter,
                                                               ctx);

  // Custom linalg-based patterns
  patterns.add<RemainderOpConversionPattern>(typeConverter, ctx);
}

void populateTTIRToTosaEltwiseBinaryPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter) {
  // Elementwise binary operations (1:1 TOSA mappings)
  patterns.add<TosaElementwiseBinaryOpConversionPattern<ttir::MinimumOp,
                                                        tosa::MinimumOp>,
               TosaElementwiseBinaryOpConversionPattern<ttir::MaximumOp,
                                                        tosa::MaximumOp>>(
      typeConverter, ctx);

  // Comparison operations
  patterns.add<
      DirectComparisonOpConversionPattern<ttir::EqualOp, tosa::EqualOp>,
      DirectComparisonOpConversionPattern<ttir::GreaterThanOp, tosa::GreaterOp>,
      DirectComparisonOpConversionPattern<ttir::GreaterEqualOp,
                                          tosa::GreaterEqualOp>,
      SwappedComparisonOpConversionPattern<ttir::LessThanOp, tosa::GreaterOp>,
      SwappedComparisonOpConversionPattern<ttir::LessEqualOp,
                                           tosa::GreaterEqualOp>,
      NegatedComparisonOpConversionPattern<ttir::NotEqualOp, tosa::EqualOp>>(
      typeConverter, ctx);

  // Logical binary operations
  patterns.add<
      LogicalBinaryOpConversionPattern<ttir::LogicalAndOp, tosa::LogicalAndOp>,
      LogicalBinaryOpConversionPattern<ttir::LogicalOrOp, tosa::LogicalOrOp>,
      LogicalBinaryOpConversionPattern<ttir::LogicalXorOp, tosa::LogicalXorOp>>(
      typeConverter, ctx);

  // Compound binary operations (multi-op TOSA sequences)
  patterns.add<GeluBackwardOpConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt::ttir_to_linalg

// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTNN/D2MToTTNN.h"

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt {

namespace {

// Helper struct to extract and return both IO and CB from a d2m.generic
// operand.
struct IOAndCB {
  Value io;
  Value cb;
};

static ttnn::ComputeKernelMathFidelity
convertMathFidelity(ttmetal::MathFidelity fidelity) {
  switch (fidelity) {
  case ttmetal::MathFidelity::LoFi:
    return ttnn::ComputeKernelMathFidelity::LoFi;
  case ttmetal::MathFidelity::HiFi2:
    return ttnn::ComputeKernelMathFidelity::HiFi2;
  case ttmetal::MathFidelity::HiFi3:
    return ttnn::ComputeKernelMathFidelity::HiFi3;
  case ttmetal::MathFidelity::HiFi4:
    return ttnn::ComputeKernelMathFidelity::HiFi4;
  }
  llvm_unreachable("Invalid MathFidelity");
}

class D2MGenericRewriter : public OpConversionPattern<d2m::GenericOp> {
public:
  D2MGenericRewriter(MLIRContext *context, ttmetal::MathFidelity mathFidelity)
      : OpConversionPattern<d2m::GenericOp>(context),
        mathFidelity(mathFidelity) {}

  static mlir::Attribute convertKernelArg(
      Builder &builder, const ttkernel::ArgAttr &arg,
      std::optional<ArrayRef<size_t>> operandToGlobalIOIndex = std::nullopt,
      size_t cbOffset = 0, size_t semaphoreOffset = 0) {
    switch (arg.getArgType()) {
    case ttkernel::ArgType::BufferAddress: {
      size_t idx = operandToGlobalIOIndex
                       ? (*operandToGlobalIOIndex)[arg.getOperandIndex()]
                       : arg.getOperandIndex();
      return builder.getAttr<ttnn::KernelArgAddressOfTensorAttr>(idx);
    }
    case ttkernel::ArgType::CBPort: {
      return builder.getAttr<ttnn::KernelArgCBBufferIndexAttr>(
          arg.getOperandIndex() + cbOffset);
    }
    case ttkernel::ArgType::Semaphore: {
      return builder.getAttr<ttnn::KernelArgSemaphoreAtAttr>(
          arg.getOperandIndex() + semaphoreOffset);
    }
    case ttkernel::ArgType::NamedArgument: {
      return builder.getAttr<ttnn::KernelArgNamedArgAttr>(
          arg.getArgumentName(), arg.getOperandIndex());
    }
    case ttkernel::ArgType::GlobalSemaphore: {
      return builder.getAttr<ttnn::KernelArgGlobalSemaphoreAttr>(
          arg.getOperandIndex());
    }
    }
  }

  static SmallVector<ttnn::KernelSemaphoreAttr>
  createSemaphoreDescriptors(Builder &builder, const ArrayAttr &threads,
                             const ttnn::CoreRangeSetAttr &coreRangeSet,
                             const SymbolTable &symbolTable,
                             size_t semaphoreStartIndex = 0) {
    llvm::DenseSet<size_t> seenSemaphoreIndices;

    for (Attribute threadAttr : threads) {
      auto thread = mlir::cast<d2m::ThreadAttr>(threadAttr);
      auto kernelFunc = symbolTable.lookup<func::FuncOp>(
          thread.getKernelSymbol().getRootReference());
      if (!kernelFunc) {
        continue;
      }

      auto kernelSpec = kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
          ttkernel::ArgSpecAttr::name);
      if (!kernelSpec) {
        continue;
      }

      for (auto ctArg : kernelSpec.getCtArgs()) {
        if (ctArg.getArgType() == ttkernel::ArgType::Semaphore) {
          seenSemaphoreIndices.insert(ctArg.getOperandIndex());
        }
      }
    }
    size_t numSemaphores = seenSemaphoreIndices.size();
    if (numSemaphores > 0) {
      // Semaphore indices are assigned sequentially in D2MToTTKernel, so they
      // should be dense.
      size_t minIndex = *llvm::min_element(seenSemaphoreIndices);
      size_t maxIndex = *llvm::max_element(seenSemaphoreIndices);
      TT_assertv((minIndex == 0u && maxIndex == numSemaphores - 1),
                 "Semaphore indices must be dense (0, 1, 2, ..., n-1)");
    }
    SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors(numSemaphores);
    for (size_t i = 0; i < numSemaphores; ++i) {
      semaphoreDescriptors[i] = builder.getAttr<ttnn::KernelSemaphoreAttr>(
          /*id=*/semaphoreStartIndex + i, ttnn::KernelCoreType::Worker,
          coreRangeSet, /*initial_value=*/0);
    }

    return semaphoreDescriptors;
  }

  static SmallVector<mlir::Attribute> createKernelDescriptors(
      Builder &builder, const ArrayAttr &threads,
      const ttnn::CoreRangeSetAttr &coreRangeSet,
      const SymbolTable &symbolTable, ttmetal::MathFidelity mathFidelity,
      std::optional<ArrayRef<size_t>> operandToGlobalIOIndex = std::nullopt,
      size_t cbOffset = 0, size_t semaphoreOffset = 0) {
    SmallVector<mlir::Attribute> kernelConfigs(threads.size());
    int unassignedNocCounter = 0;
    for (const auto [i, thread] : llvm::enumerate(threads)) {
      const d2m::ThreadAttr threadAttr = mlir::cast<d2m::ThreadAttr>(thread);

      // Get kernel args.
      SymbolRefAttr kernelSymbol = threadAttr.getKernelSymbol();
      auto kernelFunc = symbolTable.lookup<mlir::func::FuncOp>(
          kernelSymbol.getRootReference());
      auto kernelSpec = kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
          ttkernel::ArgSpecAttr::name);

      // Note: D2MToTTKernel will only populate kernelSpec with rtargs in the
      // ttnn-mode, however despite the name, they are actually common runtime
      // args. TTKernel ArgSpec does not have crt field, and the normal tt-metal
      // path doesn't use rt args at all.
      auto crtArgs = kernelSpec.getRtArgs();
      auto ctArgs = kernelSpec.getCtArgs();
      llvm::SmallVector<mlir::Attribute> kernelCTArgs(ctArgs.size());
      llvm::SmallVector<mlir::Attribute> kernelCRTArgs(crtArgs.size());
      for (const auto [j, arg] : llvm::enumerate(crtArgs)) {
        kernelCRTArgs[j] = convertKernelArg(
            builder, arg, operandToGlobalIOIndex, cbOffset, semaphoreOffset);
      }
      for (const auto [j, arg] : llvm::enumerate(ctArgs)) {
        kernelCTArgs[j] = convertKernelArg(builder, arg, operandToGlobalIOIndex,
                                           cbOffset, semaphoreOffset);
      }

      // Create KernelDescriptor.
      switch (threadAttr.getThreadType()) {
      case d2m::ThreadType::Compute: {
        // TODO (vtangTT) #5032: support lowering to different compute configs.
        kernelConfigs[i] = builder.getAttr<ttnn::ComputeKernelAttr>(
            kernelSymbol, coreRangeSet,
            /*math_fidelity*/ convertMathFidelity(mathFidelity),
            /*fp32DestAccum*/ false,
            /*dst_full_sync_en*/ false,
            /*unpack_to_dest_mode*/
            ArrayRef<ttnn::ComputeKernelUnpackToDestMode>{
                ttnn::ComputeKernelUnpackToDestMode::Default},
            /*bfp8_pack_precise*/ false,
            /*math_approx_mode*/ false, kernelCRTArgs, kernelCTArgs);
        break;
      }
      case d2m::ThreadType::Datamovement: {
        int32_t nocIdx = threadAttr.getNocIndex();
        // For unassigned NOCs, alternate between NOC0 and NOC1.
        if (nocIdx < 0) {
          nocIdx = unassignedNocCounter++ % 2;
        }
        auto nocIndex =
            nocIdx == 0 ? ttnn::NocIndex::Noc0 : ttnn::NocIndex::Noc1;
        auto processor = nocIdx == 0 ? ttnn::DataMovementProcessor::RiscV1
                                     : ttnn::DataMovementProcessor::RiscV0;
        kernelConfigs[i] = builder.getAttr<ttnn::DataMovementKernelAttr>(
            kernelSymbol, coreRangeSet, processor, nocIndex,
            ttnn::NocMode::DedicatedNoc, kernelCRTArgs, kernelCTArgs);
        break;
      }
      case d2m::ThreadType::Unified: {
        // Unified threads should have been split by SplitUnifiedThread before
        // reaching this pass.
        llvm_unreachable("Unexpected thread type in backend conversion");
      }
      }
    }
    return kernelConfigs;
  }

  static SmallVector<ttnn::KernelCBAttr> createCBDescriptors(
      Builder &builder, const llvm::SmallVector<Value> &cbs,
      const ttcore::DeviceAttr &device,
      const ttnn::CoreRangeSetAttr &coreRangeSet, size_t cbStartIndex = 0,
      std::optional<ArrayRef<size_t>> globalIOIndexForEachCB = std::nullopt) {
    if (cbs.empty()) {
      llvm_unreachable("Expected circular buffers.");
    }

    MLIRContext *ctx = builder.getContext();
    llvm::SmallVector<ttnn::KernelCBAttr> cbDescriptors(cbs.size());

    for (auto [i, cb] : llvm::enumerate(cbs)) {
      auto cb_memref = dyn_cast<MemRefType>(cb.getType());
      TT_assertv(mlir::isa<ttcore::TileType>(cb_memref.getElementType()),
                 "Only TileType supported.");
      ttcore::DataType dtype =
          ttcore::elementTypeToDataType(cb_memref.getElementType());
      size_t pageSize = device.getMemrefCBPageSizeBytes(cb_memref);
      size_t totalSize = device.getMemrefSizeBytes(cb_memref, pageSize, true);

      ttnn::KernelCBFormatAttr cbFormat =
          ttnn::KernelCBFormatAttr::get(ctx, cbStartIndex + i, dtype, pageSize);

      ttnn::KernelCBGlobalBufferAddressOfTensorAttr globalCBIndexOfTensor;
      if (mlir::isa_and_present<ttir::TTNNMetalLayoutCastOp>(
              cb.getDefiningOp()) &&
          ttcore::getMemorySpace(cb_memref) !=
              ttcore::MemorySpace::DeviceDRAM) {
        size_t tensorIndex =
            globalIOIndexForEachCB ? (*globalIOIndexForEachCB)[i] : i;
        globalCBIndexOfTensor =
            ttnn::KernelCBGlobalBufferAddressOfTensorAttr::get(ctx,
                                                               tensorIndex);
      }
      cbDescriptors[i] = ttnn::KernelCBAttr::get(
          ctx, totalSize, coreRangeSet, {cbFormat}, globalCBIndexOfTensor);
    }

    return cbDescriptors;
  }

  static IOAndCB extractIOAndCBFromGenericOperand(Value operand) {
    if (auto streamLayoutOp = mlir::dyn_cast_if_present<d2m::StreamLayoutOp>(
            operand.getDefiningOp())) {
      auto castOp = mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
          streamLayoutOp.getInput().getDefiningOp());
      TT_assertv(castOp,
                 "Expected TTNNMetalLayoutCastOp producing stream input.");
      return {castOp.getOperand(), streamLayoutOp.getStorage()};
    }

    if (auto castOp = mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
            operand.getDefiningOp())) {
      return {castOp.getOperand(), operand};
    }

    if (auto viewOp = mlir::dyn_cast_if_present<d2m::ViewLayoutOp>(
            operand.getDefiningOp())) {
      if (auto castOp = mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
              viewOp.getInput().getDefiningOp())) {
        TT_assertv(castOp,
                   "Expected TTNNMetalLayoutCastOp producing view input.");
        return {castOp.getOperand(), operand};
      }
      if (auto streamLayoutOp = mlir::dyn_cast_if_present<d2m::StreamLayoutOp>(
              viewOp.getInput().getDefiningOp())) {
        auto innerCastOp =
            mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
                streamLayoutOp.getInput().getDefiningOp());
        TT_assertv(innerCastOp,
                   "Expected TTNNMetalLayoutCastOp producing stream input.");
        return {innerCastOp.getOperand(), viewOp.getInput()};
      }
    }

    llvm_unreachable(
        "Expected stream_layout, view_layout, or cast op as operand.");
  }

  LogicalResult
  matchAndRewrite(d2m::GenericOp op, d2m::GenericOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op->getParentRegion() &&
        isa<d2m::SpatialOp>(op->getParentRegion()->getParentOp())) {
      return rewriter.notifyMatchFailure(op, "generic is inside spatial");
    }

    MLIRContext *ctx = rewriter.getContext();
    const size_t size = op.getOperands().size();
    auto device = ttcore::lookupDevice(op->getParentOp());
    TT_assert(device);

    ttcore::GridAttr opGrid = op.getGrid();
    llvm::SmallVector<int64_t> endCoreRange;
    if (!opGrid.getMapping().isEmpty()) {
      // The genericOp has a virtual grid. We need to recover the original
      // physical grid.
      auto output = op.getOutputs()[0];
      mlir::ShapedType outputType =
          mlir::cast<mlir::ShapedType>(output.getType());
      auto shardLayout = mlir::dyn_cast<ttcore::ShardLayoutAttr>(
          ttcore::getDeviceLayout(outputType));
      TT_assertv(shardLayout, "Expected shardLayoutAttr for the output of a "
                              "generic op with a virtual grid.");

      auto physicalGridShape = d2m::utils::getPhysicalGridShape(output);
      // TTNN grids are (Width, Height), while D2M grids are (Height, Width).
      endCoreRange = {physicalGridShape[1] - 1, physicalGridShape[0] - 1};
    } else {
      // TTNN grids are (Width, Height), while D2M grids are (Height, Width).
      endCoreRange = {opGrid.getShape()[1] - 1, opGrid.getShape()[0] - 1};
    }

    ttnn::CoreRangeSetAttr coreRangeSet = ttnn::CoreRangeSetAttr::get(
        ctx,
        ttnn::CoreRangeAttr::get(
            ctx, ttnn::CoreCoordAttr::get(ctx, 0, 0),
            ttnn::CoreCoordAttr::get(ctx, endCoreRange[0], endCoreRange[1])));

    llvm::SmallVector<Value> ios(size);
    llvm::SmallVector<Value> cbs(size);
    for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
      auto [io, cb] = extractIOAndCBFromGenericOperand(operand);
      ios[i] = io;
      cbs[i] = cb;
    }

    // Create CB descriptors.
    llvm::SmallVector<ttnn::KernelCBAttr> cbDescriptors =
        createCBDescriptors(rewriter, cbs, device, coreRangeSet);

    // Create KernelDescriptors.
    SymbolTable opSymTable(op->getParentOfType<ModuleOp>());
    llvm::SmallVector<mlir::Attribute> kernelDescriptors =
        createKernelDescriptors(rewriter, op.getThreads(), coreRangeSet,
                                opSymTable, this->mathFidelity);

    // Extract semaphore descriptors from kernel functions.
    llvm::SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors =
        createSemaphoreDescriptors(rewriter, op.getThreads(), coreRangeSet,
                                   opSymTable);

    ttnn::ProgramAttr program = ttnn::ProgramAttr::get(
        ctx, kernelDescriptors, cbDescriptors, semaphoreDescriptors);

    rewriter.replaceOpWithNewOp<ttnn::GenericOp>(op, ios, program,
                                                 ttnn::MemoryConfigAttr());
    return success();
  };

private:
  ttmetal::MathFidelity mathFidelity;
};

static ttnn::CoreRangeSetAttr
ttcoreRangeToTTNNCoreRangeSet(MLIRContext *ctx, ttcore::CoreRangeAttr range) {
  auto start = range.getStartCoord();
  auto end = range.getEndCoord();
  auto startCoord = ttnn::CoreCoordAttr::get(ctx, start.getX(), start.getY());
  auto endCoord = ttnn::CoreCoordAttr::get(ctx, end.getX(), end.getY());
  auto coreRange = ttnn::CoreRangeAttr::get(ctx, startCoord, endCoord);
  return ttnn::CoreRangeSetAttr::get(ctx, coreRange);
}

static std::pair<SmallVector<Value>, ttnn::ProgramAttr>
buildUnifiedProgramFromGenerics(
    Builder &builder, d2m::SpatialOp spatialOp,
    ArrayRef<d2m::GenericOp> generics,
    ArrayRef<ttnn::CoreRangeSetAttr> regionCoreRangeSets,
    const ttcore::DeviceAttr &device, SymbolTable &symbolTable,
    ttmetal::MathFidelity mathFidelity) {
  MLIRContext *ctx = builder.getContext();
  llvm::SmallDenseMap<Value, size_t> valueToGlobalIOIndex;
  SmallVector<Value> globalIos;

  for (Value v : spatialOp.getInputs()) {
    Value io = D2MGenericRewriter::extractIOAndCBFromGenericOperand(v).io;
    if (valueToGlobalIOIndex.try_emplace(io, globalIos.size()).second) {
      globalIos.push_back(io);
    }
  }
  for (Value v : spatialOp.getOutputs()) {
    Value io = D2MGenericRewriter::extractIOAndCBFromGenericOperand(v).io;
    if (valueToGlobalIOIndex.try_emplace(io, globalIos.size()).second) {
      globalIos.push_back(io);
    }
  }

  SmallVector<SmallVector<size_t>> perGenericIOIndices(generics.size());
  for (const auto [gIdx, g] : llvm::enumerate(generics)) {
    auto &indices = perGenericIOIndices[gIdx];
    auto operands = g->getOperands();
    indices.reserve(operands.size());
    for (Value operand : operands) {
      Value io =
          D2MGenericRewriter::extractIOAndCBFromGenericOperand(operand).io;
      indices.push_back(valueToGlobalIOIndex.find(io)->second);
    }
  }

  SmallVector<ttnn::KernelCBAttr> allCBs;
  SmallVector<ttnn::KernelSemaphoreAttr> allSemaphores;
  SmallVector<Attribute> allKernels;
  size_t cbNext = 0;
  size_t semNext = 0;

  for (const auto [gIdx, g] : llvm::enumerate(generics)) {
    auto operands = g->getOperands();
    llvm::SmallVector<Value> ios(operands.size());
    llvm::SmallVector<Value> cbs(operands.size());
    for (auto [i, operand] : llvm::enumerate(operands)) {
      auto [io, cb] =
          D2MGenericRewriter::extractIOAndCBFromGenericOperand(operand);
      ios[i] = io;
      cbs[i] = cb;
    }

    size_t cbOffset = cbNext;
    ArrayRef<size_t> globalIOForCBs = perGenericIOIndices[gIdx];
    auto cbDescs = D2MGenericRewriter::createCBDescriptors(
        builder, cbs, device, regionCoreRangeSets[gIdx], cbOffset,
        globalIOForCBs);
    allCBs.append(cbDescs.begin(), cbDescs.end());
    cbNext = allCBs.size();

    size_t semOffset = semNext;
    auto threads = g->getAttrOfType<ArrayAttr>("threads");
    auto semDescs = D2MGenericRewriter::createSemaphoreDescriptors(
        builder, threads, regionCoreRangeSets[gIdx], symbolTable, semOffset);
    allSemaphores.append(semDescs.begin(), semDescs.end());
    semNext = allSemaphores.size();

    ArrayRef<size_t> ioMap = perGenericIOIndices[gIdx];
    auto kernelDescs = D2MGenericRewriter::createKernelDescriptors(
        builder, threads, regionCoreRangeSets[gIdx], symbolTable, mathFidelity,
        ioMap, cbOffset, semOffset);
    allKernels.append(kernelDescs.begin(), kernelDescs.end());
  }

  ttnn::ProgramAttr program =
      ttnn::ProgramAttr::get(ctx, allKernels, allCBs, allSemaphores);
  return {std::move(globalIos), program};
}

class D2MSpatialRewriter : public OpConversionPattern<d2m::SpatialOp> {
public:
  D2MSpatialRewriter(MLIRContext *context, ttmetal::MathFidelity mathFidelity)
      : OpConversionPattern<d2m::SpatialOp>(context),
        mathFidelity(mathFidelity) {}

  LogicalResult
  matchAndRewrite(d2m::SpatialOp op, d2m::SpatialOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getNumRegions() == 0) {
      return rewriter.notifyMatchFailure(op, "no regions");
    }

    SmallVector<d2m::GenericOp> generics;
    generics.reserve(op.getNumRegions());
    for (Region &r : op.getRegions()) {
      d2m::GenericOp generic;
      r.walk([&](d2m::GenericOp g) {
        if (!generic) {
          generic = g;
        }
      });
      if (!generic) {
        return rewriter.notifyMatchFailure(op, "region has no generic");
      }
      generics.push_back(generic);
    }

    auto device = ttcore::lookupDevice(op->getParentOp());
    if (!device) {
      return rewriter.notifyMatchFailure(op, "no device");
    }

    auto gridRanges = op.getGridRanges().getCoreRanges();
    SmallVector<ttnn::CoreRangeSetAttr> regionCoreRangeSets;
    regionCoreRangeSets.reserve(gridRanges.size());
    for (ttcore::CoreRangeAttr range : gridRanges) {
      regionCoreRangeSets.push_back(
          ttcoreRangeToTTNNCoreRangeSet(rewriter.getContext(), range));
    }

    SymbolTable symbolTable(op->getParentOfType<ModuleOp>());
    auto [globalIos, program] = buildUnifiedProgramFromGenerics(
        rewriter, op, generics, regionCoreRangeSets, device, symbolTable,
        mathFidelity);

    rewriter.replaceOpWithNewOp<ttnn::GenericOp>(op, globalIos, program,
                                                 ttnn::MemoryConfigAttr());
    return success();
  }

private:
  ttmetal::MathFidelity mathFidelity;
};
} // namespace

namespace {
class TTNNMetalLayoutCastRewriter
    : public OpConversionPattern<ttir::TTNNMetalLayoutCastOp> {
public:
  using OpConversionPattern<ttir::TTNNMetalLayoutCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::TTNNMetalLayoutCastOp op,
                  ttir::TTNNMetalLayoutCastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (auto inner =
            op.getOperand().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
      // At this point (D2M→TTNN conversion), the D2M pipeline has already
      // materialized all data movement for VGMs. Back-to-back casts can
      // be safely collapsed even when they carry VGM attributes.
      rewriter.replaceOp(op, inner.getOperand());
    } else if (auto inner =
                   op.getOperand().getDefiningOp<d2m::StreamLayoutOp>()) {
      // Match the pattern cast(stream(cast(output_tensor))) and rewrite as just
      // output_tensor.
      if (auto inner2 =
              inner.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
        rewriter.replaceOp(op, inner2.getOperand());
      }
    } else if (auto inner =
                   op.getOperand().getDefiningOp<d2m::ViewLayoutOp>()) {
      // Match the pattern cast(view(cast(output_tensor))) and rewrite as just
      // output_tensor.
      if (auto inner2 =
              inner.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
        rewriter.replaceOp(op, inner2.getOperand());
      }
    }
    return success();
  };
};
} // namespace

namespace {
class StreamLayoutRewriter : public OpConversionPattern<d2m::StreamLayoutOp> {
public:
  using OpConversionPattern<d2m::StreamLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::StreamLayoutOp op, d2m::StreamLayoutOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  };
};
} // namespace

namespace {
class ViewLayoutRewriter : public OpConversionPattern<d2m::ViewLayoutOp> {
public:
  using OpConversionPattern<d2m::ViewLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::ViewLayoutOp op, d2m::ViewLayoutOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  };
};
} // namespace

namespace {
class D2MEmptyRewriter : public OpConversionPattern<d2m::EmptyOp> {
public:
  using OpConversionPattern<d2m::EmptyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::EmptyOp op, d2m::EmptyOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *ctx = rewriter.getContext();
    auto tensorType = cast<RankedTensorType>(op.getResult().getType());
    auto encoding = tensorType.getEncoding();
    auto shape = ttnn::ShapeAttr::get(ctx, tensorType.getShape());

    ttcore::DataTypeAttr dtype;
    ttnn::LayoutAttr layout;
    ttnn::MemoryConfigAttr memcfg;

    // Reuses the existing ttnn.get_device op if present, else create one.
    auto device = ttnn::utils::getOrInsertDevice(rewriter, op);
    auto deviceAttr = ttcore::lookupDevice(op);

    // Handle both TTNNLayoutAttr and TTNNNDLayoutAttr
    if (auto layoutAttr = mlir::dyn_cast<ttnn::TTNNLayoutAttr>(encoding)) {
      dtype = ttcore::DataTypeAttr::get(ctx, layoutAttr.getDataType());
      layout = ttnn::LayoutAttr::get(ctx, layoutAttr.getLayout());
      memcfg =
          ttnn::MemoryConfigAttr::get(layoutAttr, deviceAttr.getWorkerGrid());
    } else if (auto ndLayoutAttr =
                   mlir::dyn_cast<ttnn::TTNNNDLayoutAttr>(encoding)) {
      dtype = ttcore::DataTypeAttr::get(ctx, ndLayoutAttr.getDataType());
      layout = ttnn::LayoutAttr::get(ctx, ndLayoutAttr.getLayout());
      auto bufferType =
          ttnn::BufferTypeAttr::get(ctx, ndLayoutAttr.getBufferType());
      auto ndShardSpec = ttnn::NDShardSpecAttr::get(ndLayoutAttr);
      memcfg = ttnn::MemoryConfigAttr::get(
          ctx, ndLayoutAttr.getMemLayout(), bufferType,
          /*shardSpec=*/std::nullopt, ndShardSpec);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported encoding type");
    }

    rewriter.replaceOpWithNewOp<ttnn::EmptyOp>(op, tensorType, device, shape,
                                               dtype, layout, memcfg);
    return success();
  };
};
} // namespace

namespace {
class D2MFullRewriter : public OpConversionPattern<d2m::FullOp> {
public:
  using OpConversionPattern<d2m::FullOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::FullOp op, d2m::FullOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *ctx = rewriter.getContext();
    auto tensorType = cast<RankedTensorType>(op.getResult().getType());
    auto encoding = tensorType.getEncoding();

    // Convert DenseI32ArrayAttr shape to ttnn::ShapeAttr
    auto shapeI32 = adaptor.getShape();
    SmallVector<int64_t> shapeI64(shapeI32.begin(), shapeI32.end());
    auto shape = ttnn::ShapeAttr::get(ctx, shapeI64);

    ttcore::DataTypeAttr dtype;
    ttnn::LayoutAttr layout;
    ttnn::MemoryConfigAttr memcfg;

    // Reuses the existing ttnn.get_device op if present, else create one.
    auto device = ttnn::utils::getOrInsertDevice(rewriter, op);
    auto deviceAttr = ttcore::lookupDevice(op);

    // Handle both TTNNLayoutAttr and TTNNNDLayoutAttr
    if (auto layoutAttr = mlir::dyn_cast<ttnn::TTNNLayoutAttr>(encoding)) {
      dtype = ttcore::DataTypeAttr::get(ctx, layoutAttr.getDataType());
      layout = ttnn::LayoutAttr::get(ctx, layoutAttr.getLayout());
      memcfg =
          ttnn::MemoryConfigAttr::get(layoutAttr, deviceAttr.getWorkerGrid());
    } else if (auto ndLayoutAttr =
                   mlir::dyn_cast<ttnn::TTNNNDLayoutAttr>(encoding)) {
      dtype = ttcore::DataTypeAttr::get(ctx, ndLayoutAttr.getDataType());
      layout = ttnn::LayoutAttr::get(ctx, ndLayoutAttr.getLayout());
      auto bufferType =
          ttnn::BufferTypeAttr::get(ctx, ndLayoutAttr.getBufferType());
      auto ndShardSpec = ttnn::NDShardSpecAttr::get(ndLayoutAttr);
      memcfg = ttnn::MemoryConfigAttr::get(
          ctx, ndLayoutAttr.getMemLayout(), bufferType,
          /*shardSpec=*/std::nullopt, ndShardSpec);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported encoding type");
    }

    rewriter.replaceOpWithNewOp<ttnn::FullOp>(op, tensorType, device, shape,
                                              adaptor.getFillValue(), dtype,
                                              layout, memcfg);
    return success();
  };
};
} // namespace

void populateD2MToTTNNPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                               TypeConverter &typeConverter,
                               ttmetal::MathFidelity mathFidelity) {
  patterns.add<D2MSpatialRewriter>(ctx, mathFidelity);
  patterns.add<D2MGenericRewriter>(ctx, mathFidelity);
  patterns.add<TTNNMetalLayoutCastRewriter, D2MEmptyRewriter, D2MFullRewriter,
               StreamLayoutRewriter, ViewLayoutRewriter>(ctx);
}
} // namespace mlir::tt

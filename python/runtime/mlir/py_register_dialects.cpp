/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/InitAllPasses.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/InitAllDialects.h"
#include <fmt/core.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace mlir::python::adaptors;
using namespace mlir;

namespace cudaq {
static bool registered = false;

void registerQuakeDialectAndTypes(py::module &m) {
  auto quakeMod = m.def_submodule("quake");

  quakeMod.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__quake__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load)
          mlirDialectHandleLoadDialect(handle, context);

        if (!registered) {
          cudaq::registerCudaqPassesAndPipelines();
          registered = true;
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  mlir_type_subclass(quakeMod, "RefType", [](MlirType type) {
    return unwrap(type).isa<quake::RefType>();
  }).def_classmethod("get", [](py::object cls, MlirContext ctx) {
    return wrap(quake::RefType::get(unwrap(ctx)));
  });

  mlir_type_subclass(quakeMod, "MeasureType", [](MlirType type) {
    return unwrap(type).isa<quake::MeasureType>();
  }).def_classmethod("get", [](py::object cls, MlirContext ctx) {
    return wrap(quake::MeasureType::get(unwrap(ctx)));
  });

  mlir_type_subclass(
      quakeMod, "VeqType",
      [](MlirType type) { return unwrap(type).isa<quake::VeqType>(); })
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx, std::size_t size) {
            return wrap(quake::VeqType::get(unwrap(ctx), size));
          },
          py::arg("cls"), py::arg("context"),
          py::arg("size") = std::numeric_limits<std::size_t>::max())
      .def_staticmethod(
          "hasSpecifiedSize",
          [](MlirType type) {
            auto veqTy = dyn_cast<quake::VeqType>(unwrap(type));
            if (!veqTy)
              throw std::runtime_error(
                  "Invalid type passed to VeqType.getSize()");

            return veqTy.hasSpecifiedSize();
          },
          py::arg("veqTypeInstance"))
      .def_staticmethod(
          "getSize",
          [](MlirType type) {
            auto veqTy = dyn_cast<quake::VeqType>(unwrap(type));
            if (!veqTy)
              throw std::runtime_error(
                  "Invalid type passed to VeqType.getSize()");

            return veqTy.getSize();
          },
          py::arg("veqTypeInstance"));

  mlir_type_subclass(
      quakeMod, "StruqType",
      [](MlirType type) { return unwrap(type).isa<quake::StruqType>(); })
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx, py::list aggregateTypes) {
            SmallVector<Type> inTys;
            for (auto &t : aggregateTypes)
              inTys.push_back(unwrap(t.cast<MlirType>()));

            return wrap(quake::StruqType::get(unwrap(ctx), inTys));
          })
      .def_classmethod("getNamed",
                       [](py::object cls, MlirContext ctx,
                          const std::string &name, py::list aggregateTypes) {
                         SmallVector<Type> inTys;
                         for (auto &t : aggregateTypes)
                           inTys.push_back(unwrap(t.cast<MlirType>()));

                         return wrap(
                             quake::StruqType::get(unwrap(ctx), name, inTys));
                       })
      .def_classmethod(
          "getTypes",
          [](py::object cls, MlirType structTy) {
            auto ty = dyn_cast<quake::StruqType>(unwrap(structTy));
            if (!ty)
              throw std::runtime_error(
                  "invalid type passed to StruqType.getTypes(), must be a "
                  "quake.struq");
            std::vector<MlirType> ret;
            for (auto &t : ty.getMembers())
              ret.push_back(wrap(t));
            return ret;
          })
      .def_classmethod("getName", [](py::object cls, MlirType structTy) {
        auto ty = dyn_cast<quake::StruqType>(unwrap(structTy));
        if (!ty)
          throw std::runtime_error(
              "invalid type passed to StruqType.getName(), must be a "
              "quake.struq");
        return ty.getName().getValue().str();
      });
}

void registerCCDialectAndTypes(py::module &m) {

  auto ccMod = m.def_submodule("cc");

  ccMod.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle ccHandle = mlirGetDialectHandle__cc__();
        mlirDialectHandleRegisterDialect(ccHandle, context);
        if (load) {
          mlirDialectHandleLoadDialect(ccHandle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  mlir_type_subclass(ccMod, "CharspanType", [](MlirType type) {
    return unwrap(type).isa<cudaq::cc::CharspanType>();
  }).def_classmethod("get", [](py::object cls, MlirContext ctx) {
    return wrap(cudaq::cc::CharspanType::get(unwrap(ctx)));
  });

  mlir_type_subclass(ccMod, "StateType", [](MlirType type) {
    return unwrap(type).isa<quake::StateType>();
  }).def_classmethod("get", [](py::object cls, MlirContext ctx) {
    return wrap(quake::StateType::get(unwrap(ctx)));
  });

  mlir_type_subclass(
      ccMod, "PointerType",
      [](MlirType type) { return unwrap(type).isa<cudaq::cc::PointerType>(); })
      .def_classmethod(
          "getElementType",
          [](py::object cls, MlirType type) {
            auto ty = unwrap(type);
            auto casted = dyn_cast<cudaq::cc::PointerType>(ty);
            if (!casted)
              throw std::runtime_error(
                  "invalid type passed to PointerType.getElementType(), must "
                  "be cc.ptr type.");
            return wrap(casted.getElementType());
          })
      .def_classmethod(
          "get", [](py::object cls, MlirContext ctx, MlirType elementType) {
            return wrap(
                cudaq::cc::PointerType::get(unwrap(ctx), unwrap(elementType)));
          });

  mlir_type_subclass(
      ccMod, "ArrayType",
      [](MlirType type) { return unwrap(type).isa<cudaq::cc::ArrayType>(); })
      .def_classmethod(
          "getElementType",
          [](py::object cls, MlirType type) {
            auto ty = unwrap(type);
            auto casted = dyn_cast<cudaq::cc::ArrayType>(ty);
            if (!casted)
              throw std::runtime_error(
                  "invalid type passed to ArrayType.getElementType(), must "
                  "be cc.array type.");
            return wrap(casted.getElementType());
          })
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx, MlirType elementType,
             std::int64_t size) {
            return wrap(cudaq::cc::ArrayType::get(unwrap(ctx),
                                                  unwrap(elementType), size));
          },
          py::arg("cls"), py::arg("ctx"), py::arg("elementType"),
          py::arg("size") = std::numeric_limits<std::int64_t>::min());

  mlir_type_subclass(
      ccMod, "StructType",
      [](MlirType type) { return unwrap(type).isa<cudaq::cc::StructType>(); })
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx, py::list aggregateTypes) {
            SmallVector<Type> inTys;
            for (auto &t : aggregateTypes)
              inTys.push_back(unwrap(t.cast<MlirType>()));

            return wrap(cudaq::cc::StructType::get(unwrap(ctx), inTys));
          })
      .def_classmethod(
          "getNamed",
          [](py::object cls, MlirContext ctx, const std::string &name,
             py::list aggregateTypes) {
            SmallVector<Type> inTys;
            for (auto &t : aggregateTypes)
              inTys.push_back(unwrap(t.cast<MlirType>()));

            return wrap(cudaq::cc::StructType::get(unwrap(ctx), name, inTys));
          })
      .def_classmethod(
          "getTypes",
          [](py::object cls, MlirType structTy) {
            auto ty = dyn_cast<cudaq::cc::StructType>(unwrap(structTy));
            if (!ty)
              throw std::runtime_error(
                  "invalid type passed to StructType.getTypes(), must be a "
                  "cc.struct");
            std::vector<MlirType> ret;
            for (auto &t : ty.getMembers())
              ret.push_back(wrap(t));
            return ret;
          })
      .def_classmethod("getName", [](py::object cls, MlirType structTy) {
        auto ty = dyn_cast<cudaq::cc::StructType>(unwrap(structTy));
        if (!ty)
          throw std::runtime_error(
              "invalid type passed to StructType.getName(), must be a "
              "cc.struct");
        return ty.getName().getValue().str();
      });

  mlir_type_subclass(
      ccMod, "CallableType",
      [](MlirType type) { return unwrap(type).isa<cudaq::cc::CallableType>(); })
      .def_classmethod("get",
                       [](py::object cls, MlirContext ctx, py::list inTypes) {
                         SmallVector<Type> inTys;
                         for (auto &t : inTypes)
                           inTys.push_back(unwrap(t.cast<MlirType>()));

                         return wrap(cudaq::cc::CallableType::get(
                             unwrap(ctx), FunctionType::get(unwrap(ctx), inTys,
                                                            TypeRange{})));
                       })
      .def_classmethod("getFunctionType", [](py::object cls, MlirType type) {
        return wrap(
            dyn_cast<cudaq::cc::CallableType>(unwrap(type)).getSignature());
      });

  mlir_type_subclass(
      ccMod, "StdvecType",
      [](MlirType type) { return unwrap(type).isa<cudaq::cc::StdvecType>(); })
      .def_classmethod(
          "getElementType",
          [](py::object cls, MlirType type) {
            auto ty = unwrap(type);
            auto casted = dyn_cast<cudaq::cc::StdvecType>(ty);
            if (!casted)
              throw std::runtime_error(
                  "invalid type passed to StdvecType.getElementType(), must "
                  "be cc.array type.");
            return wrap(casted.getElementType());
          })
      .def_classmethod(
          "get", [](py::object cls, MlirContext ctx, MlirType elementType) {
            return wrap(
                cudaq::cc::StdvecType::get(unwrap(ctx), unwrap(elementType)));
          });
}

void bindRegisterDialects(py::module &mod) {
  registerQuakeDialectAndTypes(mod);
  registerCCDialectAndTypes(mod);

  mod.def("load_intrinsic", [](MlirModule module, std::string name) {
    auto unwrapped = unwrap(module);
    cudaq::IRBuilder builder = IRBuilder::atBlockEnd(unwrapped.getBody());
    if (failed(builder.loadIntrinsic(unwrapped, name)))
      unwrapped.emitError("failed to load intrinsic " + name);
  });

  mod.def("register_all_dialects", [](MlirContext context) {
    DialectRegistry registry;
    registry.insert<quake::QuakeDialect, cudaq::cc::CCDialect>();
    cudaq::opt::registerCodeGenDialect(registry);
    registerAllDialects(registry);
    auto *mlirContext = unwrap(context);
    mlirContext->appendDialectRegistry(registry);
    mlirContext->loadAllAvailableDialects();
  });

  mod.def("gen_vector_of_complex_constant", [](MlirLocation loc,
                                               MlirModule module,
                                               std::string name,
                                               const std::vector<std::complex<
                                                   double>> &values) {
    ModuleOp modOp = unwrap(module);
    cudaq::IRBuilder builder = IRBuilder::atBlockEnd(modOp.getBody());
    SmallVector<std::complex<double>> newValues{values.begin(), values.end()};
    builder.genVectorOfConstants(unwrap(loc), modOp, name, newValues);
  });
}
} // namespace cudaq

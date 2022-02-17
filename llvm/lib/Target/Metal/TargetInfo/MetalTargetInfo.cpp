//===-- MetalTargetInfo.cpp - AIR Target Implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../MetalTargetMachine.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

Target &llvm::getTheMetalTarget() {
  static Target TheMetalTarget;
  return TheMetalTarget;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeMetalTargetInfo() {
  RegisterTarget<Triple::air64> X(getTheMetalTarget(), "metal", "Metal", "Metal");
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeMetalTargetMC() {}
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeMetalAsmPrinter() {}
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeMetalAsmParser() {}


//===- Metal.h - Top-level interface for Metal back-end--------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// The original implementation is from libfloor (https://github.com/a2flo/floor)
// Copyright (C) 2004 - 2021 Florian Ziesche
//
//===----------------------------------------------------------------------===//

#include "MetalTargetMachine.h"

#include "llvm/Pass.h"

namespace llvm {

//===----------------------------------------------------------------------===//
//
// MetalFirst - This pass fixes Metal/AIR issues.
//
FunctionPass *createMetalFirstPass(const bool enable_intel_workarounds = false,
                                   const bool enable_nvidia_workarounds = false);
void initializeMetalFirstPass(PassRegistry&);

//===----------------------------------------------------------------------===//
//
// MetalFinal - This pass fixes Metal/AIR issues.
//
FunctionPass *createMetalFinalPass(const bool enable_intel_workarounds = false,
                                   const bool enable_nvidia_workarounds = false);
void initializeMetalFinalPass(PassRegistry&);

//===----------------------------------------------------------------------===//
//
// MetalFinalModuleCleanup - This pass removes any calling convention attributes
// and removes unused functions/prototypes/externs.
//
ModulePass *createMetalFinalModuleCleanupPass();
void initializeMetalFinalModuleCleanupPass(PassRegistry&);

ModulePass *createMetalLibWriterPass(raw_ostream &Str);

}

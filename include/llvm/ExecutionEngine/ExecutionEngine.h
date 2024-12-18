//===- ExecutionEngine.h - Abstract Execution Engine Interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the abstract interface that implements execution support
// for LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef EXECUTION_ENGINE_H
#define EXECUTION_ENGINE_H

#include <vector>
#include <string>
#include <map>
#include <cassert>
class Constant;
class Function;
union GenericValue;
class GlobalValue;
class Module;
class ModuleProvider;
class TargetData;
class Type;

class ExecutionEngine {
  Module &CurMod;
  const TargetData *TD;

protected:
  ModuleProvider *MP;  // OWN it
  // GlobalAddress - A mapping between LLVM global values and their actualized
  // version...
  std::map<const GlobalValue*, void *> GlobalAddress;

  void setTargetData(const TargetData &td) {
    TD = &td;
  }

public:
  ExecutionEngine(ModuleProvider *P);
  ExecutionEngine(Module *M);
  virtual ~ExecutionEngine();

  Module &getModule() const { return CurMod; }
  const TargetData &getTargetData() const { return *TD; }

  /// run - Start execution with the specified function, arguments, and
  ///       environment.
  ///
  virtual GenericValue run(Function *F,
                           const std::vector<GenericValue> &ArgValues) = 0;

  static ExecutionEngine *create(ModuleProvider *MP, bool ForceInterpreter,
                                 bool TraceMode);

  void addGlobalMapping(const Function *F, void *Addr) {
    void *&CurVal = GlobalAddress[(const GlobalValue*)F];
    assert(CurVal == 0 && "GlobalMapping already established!");
    CurVal = Addr;
  }

  // getPointerToGlobalIfAvailable - This returns the address of the specified
  // global value if it is available, otherwise it returns null.
  //
  void *getPointerToGlobalIfAvailable(const GlobalValue *GV) {
    std::map<const GlobalValue*, void*>::iterator I = GlobalAddress.find(GV);
    return I != GlobalAddress.end() ? I->second : 0;
  }

  // getPointerToGlobal - This returns the address of the specified global
  // value.  This may involve code generation if it's a function.
  //
  void *getPointerToGlobal(const GlobalValue *GV);

  // getPointerToFunction - The different EE's represent function bodies in
  // different ways.  They should each implement this to say what a function
  // pointer should look like.
  //
  virtual void *getPointerToFunction(Function *F) = 0;

  void StoreValueToMemory(GenericValue Val, GenericValue *Ptr, const Type *Ty);
  void InitializeMemory(const Constant *Init, void *Addr);

protected:
  void emitGlobals();
  GenericValue getConstantValue(const Constant *C);
  GenericValue LoadValueFromMemory(GenericValue *Ptr, const Type *Ty);
};

#endif

//===- Support/Signals.h - Signal Handling support --------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines some helpful functions for dealing with the possibility of
// unix signals occuring while your program is running.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_SIGNALS_H
#define SUPPORT_SIGNALS_H

#include <string>

// RemoveFileOnSignal - This function registers signal handlers to ensure that
// if a signal gets delivered that the named file is removed.
//
void RemoveFileOnSignal(const std::string &Filename);

#endif


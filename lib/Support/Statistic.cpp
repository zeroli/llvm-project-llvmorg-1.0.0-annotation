//===-- Statistic.cpp - Easy way to expose stats information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'Statistic' class, which is designed to be an easy
// way to expose various success metrics from passes.  These statistics are
// printed at the end of a run, when the -stats command line option is enabled
// on the command line.
//
// This is useful for reporting information like the number of instructions
// simplified, optimized or removed by various transformations, like this:
//
// static Statistic<> NumInstEliminated("GCSE - Number of instructions killed");
//
// Later, in the code: ++NumInstEliminated;
//
//===----------------------------------------------------------------------===//

#include "Support/Statistic.h"
#include "Support/CommandLine.h"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstring>

// GetLibSupportInfoOutputFile - Return a file stream to print our output on...
std::ostream *GetLibSupportInfoOutputFile();

unsigned StatisticBase::NumStats = 0;

// -stats - Command line option to cause transformations to emit stats about
// what they did.
//
static cl::opt<bool>
Enabled("stats", cl::desc("Enable statistics output from program"));

struct StatRecord {
  std::string Value;
  const char *Name, *Desc;

  StatRecord(const std::string V, const char *N, const char *D)
    : Value(V), Name(N), Desc(D) {}

  bool operator<(const StatRecord &SR) const {
    return std::strcmp(Name, SR.Name) < 0;
  }

  void print(unsigned ValFieldSize, unsigned NameFieldSize,
             std::ostream &OS) {
    OS << std::string(ValFieldSize-Value.length(), ' ')
       << Value << " " << Name
       << std::string(NameFieldSize-std::strlen(Name), ' ')
       << " - " << Desc << "\n";
  }
};

static std::vector<StatRecord> *AccumStats = 0;

// Print information when destroyed, iff command line option is specified
void StatisticBase::destroy() const {
  if (Enabled && hasSomeData()) {
    if (AccumStats == 0)
      AccumStats = new std::vector<StatRecord>();

    std::ostringstream Out;
    printValue(Out);
    AccumStats->push_back(StatRecord(Out.str(), Name, Desc));
  }

  if (--NumStats == 0 && AccumStats) {
    std::ostream *OutStream = GetLibSupportInfoOutputFile();

    // Figure out how long the biggest Value and Name fields are...
    unsigned MaxNameLen = 0, MaxValLen = 0;
    for (unsigned i = 0, e = AccumStats->size(); i != e; ++i) {
      MaxValLen = std::max(MaxValLen,
                           (unsigned)(*AccumStats)[i].Value.length());
      MaxNameLen = std::max(MaxNameLen,
                            (unsigned)std::strlen((*AccumStats)[i].Name));
    }

    // Sort the fields...
    std::stable_sort(AccumStats->begin(), AccumStats->end());

    // Print out the statistics header...
    *OutStream << "===" << std::string(73, '-') << "===\n"
               << "                          ... Statistics Collected ...\n"
               << "===" << std::string(73, '-') << "===\n\n";

    // Print all of the statistics accumulated...
    for (unsigned i = 0, e = AccumStats->size(); i != e; ++i)
      (*AccumStats)[i].print(MaxValLen, MaxNameLen, *OutStream);

    *OutStream << std::endl;  // Flush the output stream...

    // Free all accumulated statistics...
    delete AccumStats;
    AccumStats = 0;
    if (OutStream != &std::cerr && OutStream != &std::cout)
      delete OutStream;   // Close the file...
  }
}

//===- llvm/Analysis/Dominators.h - Dominator Info Calculation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the following classes:
//  1. DominatorSet: Calculates the [reverse] dominator set for a function
//  2. ImmediateDominators: Calculates and holds a mapping between BasicBlocks
//     and their immediate dominator.
//  3. DominatorTree: Represent the ImmediateDominator as an explicit tree
//     structure.
//  4. DominanceFrontier: Calculate and hold the dominance frontier for a
//     function.
//
//  These data structures are listed in increasing order of complexity.  It
//  takes longer to calculate the dominator frontier, for example, than the
//  ImmediateDominator mapping.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DOMINATORS_H
#define LLVM_ANALYSIS_DOMINATORS_H

#include "llvm/Pass.h"
#include <set>

class Instruction;

template <typename GraphType> struct GraphTraits;

//===----------------------------------------------------------------------===//
//
// DominatorBase - Base class that other, more interesting dominator analyses
// inherit from.
//
class DominatorBase : public FunctionPass {
protected:
  std::vector<BasicBlock*> Roots;
  const bool IsPostDominators;

  inline DominatorBase(bool isPostDom) : Roots(), IsPostDominators(isPostDom) {}
public:
  // Return the root blocks of the current CFG.  This may include multiple
  // blocks if we are computing post dominators.  For forward dominators, this
  // will always be a single block (the entry node).
  inline const std::vector<BasicBlock*> &getRoots() const { return Roots; }

  // Returns true if analysis based of postdoms
  bool isPostDominator() const { return IsPostDominators; }
};

//===----------------------------------------------------------------------===//
//
// DominatorSet - Maintain a set<BasicBlock*> for every basic block in a
// function, that represents the blocks that dominate the block.  If the block
// is unreachable in this function, the set will be empty.  This cannot happen
// for reachable code, because every block dominates at least itself.
//
struct DominatorSetBase : public DominatorBase {
  typedef std::set<BasicBlock*> DomSetType;    // Dom set for a bb
  // Map of dom sets
  typedef std::map<BasicBlock*, DomSetType> DomSetMapType;
protected:
  DomSetMapType Doms;
public:
  DominatorSetBase(bool isPostDom) : DominatorBase(isPostDom) {}

  virtual void releaseMemory() { Doms.clear(); }

  // Accessor interface:
  typedef DomSetMapType::const_iterator const_iterator;
  typedef DomSetMapType::iterator iterator;
  inline const_iterator begin() const { return Doms.begin(); }
  inline       iterator begin()       { return Doms.begin(); }
  inline const_iterator end()   const { return Doms.end(); }
  inline       iterator end()         { return Doms.end(); }
  inline const_iterator find(BasicBlock* B) const { return Doms.find(B); }
  inline       iterator find(BasicBlock* B)       { return Doms.find(B); }


  /// getDominators - Return the set of basic blocks that dominate the specified
  /// block.
  ///
  inline const DomSetType &getDominators(BasicBlock *BB) const {
    const_iterator I = find(BB);
    assert(I != end() && "BB not in function!");
    return I->second;
  }

  /// isReachable - Return true if the specified basicblock is reachable.  If
  /// the block is reachable, we have dominator set information for it.
  bool isReachable(BasicBlock *BB) const {
    return !getDominators(BB).empty();
  }

  /// dominates - Return true if A dominates B.
  ///
  inline bool dominates(BasicBlock *A, BasicBlock *B) const {
    return getDominators(B).count(A) != 0;
  }

  /// properlyDominates - Return true if A dominates B and A != B.
  ///
  bool properlyDominates(BasicBlock *A, BasicBlock *B) const {
    return dominates(A, B) && A != B;
  }

  /// print - Convert to human readable form
  virtual void print(std::ostream &OS) const;

  /// dominates - Return true if A dominates B.  This performs the special
  /// checks necessary if A and B are in the same basic block.
  ///
  bool dominates(Instruction *A, Instruction *B) const;

  //===--------------------------------------------------------------------===//
  // API to update (Post)DominatorSet information based on modifications to
  // the CFG...

  /// addBasicBlock - Call to update the dominator set with information about a
  /// new block that was inserted into the function.
  void addBasicBlock(BasicBlock *BB, const DomSetType &Dominators) {
    assert(find(BB) == end() && "Block already in DominatorSet!");
    Doms.insert(std::make_pair(BB, Dominators));
  }

  // addDominator - If a new block is inserted into the CFG, then method may be
  // called to notify the blocks it dominates that it is in their set.
  //
  void addDominator(BasicBlock *BB, BasicBlock *NewDominator) {
    iterator I = find(BB);
    assert(I != end() && "BB is not in DominatorSet!");
    I->second.insert(NewDominator);
  }
};


//===-------------------------------------
// DominatorSet Class - Concrete subclass of DominatorSetBase that is used to
// compute a normal dominator set.
//
struct DominatorSet : public DominatorSetBase {
  DominatorSet() : DominatorSetBase(false) {}

  virtual bool runOnFunction(Function &F);

  /// recalculate - This method may be called by external passes that modify the
  /// CFG and then need dominator information recalculated.  This method is
  /// obviously really slow, so it should be avoided if at all possible.
  void recalculate();

  BasicBlock *getRoot() const {
    assert(Roots.size() == 1 && "Should always have entry node!");
    return Roots[0];
  }

  // getAnalysisUsage - This simply provides a dominator set
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
private:
  void calculateDominatorsFromBlock(BasicBlock *BB);
};


//===----------------------------------------------------------------------===//
//
// ImmediateDominators - Calculate the immediate dominator for each node in a
// function.
//
class ImmediateDominatorsBase : public DominatorBase {
protected:
  std::map<BasicBlock*, BasicBlock*> IDoms;
  void calcIDoms(const DominatorSetBase &DS);
public:
  ImmediateDominatorsBase(bool isPostDom) : DominatorBase(isPostDom) {}

  virtual void releaseMemory() { IDoms.clear(); }

  // Accessor interface:
  typedef std::map<BasicBlock*, BasicBlock*> IDomMapType;
  typedef IDomMapType::const_iterator const_iterator;
  inline const_iterator begin() const { return IDoms.begin(); }
  inline const_iterator end()   const { return IDoms.end(); }
  inline const_iterator find(BasicBlock* B) const { return IDoms.find(B);}

  // operator[] - Return the idom for the specified basic block.  The start
  // node returns null, because it does not have an immediate dominator.
  //
  inline BasicBlock *operator[](BasicBlock *BB) const {
    return get(BB);
  }

  // get() - Synonym for operator[].
  inline BasicBlock *get(BasicBlock *BB) const {
    std::map<BasicBlock*, BasicBlock*>::const_iterator I = IDoms.find(BB);
    return I != IDoms.end() ? I->second : 0;
  }

  //===--------------------------------------------------------------------===//
  // API to update Immediate(Post)Dominators information based on modifications
  // to the CFG...

  /// addNewBlock - Add a new block to the CFG, with the specified immediate
  /// dominator.
  ///
  void addNewBlock(BasicBlock *BB, BasicBlock *IDom) {
    assert(get(BB) == 0 && "BasicBlock already in idom info!");
    IDoms[BB] = IDom;
  }

  /// setImmediateDominator - Update the immediate dominator information to
  /// change the current immediate dominator for the specified block to another
  /// block.  This method requires that BB already have an IDom, otherwise just
  /// use addNewBlock.
  void setImmediateDominator(BasicBlock *BB, BasicBlock *NewIDom) {
    assert(IDoms.find(BB) != IDoms.end() && "BB doesn't have idom yet!");
    IDoms[BB] = NewIDom;
  }

  // print - Convert to human readable form
  virtual void print(std::ostream &OS) const;
};

//===-------------------------------------
// ImmediateDominators Class - Concrete subclass of ImmediateDominatorsBase that
// is used to compute a normal immediate dominator set.
//
struct ImmediateDominators : public ImmediateDominatorsBase {
  ImmediateDominators() : ImmediateDominatorsBase(false) {}

  BasicBlock *getRoot() const {
    assert(Roots.size() == 1 && "Should always have entry node!");
    return Roots[0];
  }

  virtual bool runOnFunction(Function &F) {
    IDoms.clear();     // Reset from the last time we were run...
    DominatorSet &DS = getAnalysis<DominatorSet>();
    Roots = DS.getRoots();
    calcIDoms(DS);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<DominatorSet>();
  }
};


//===----------------------------------------------------------------------===//
//
// DominatorTree - Calculate the immediate dominator tree for a function.
//
struct DominatorTreeBase : public DominatorBase {
  class Node;
protected:
  std::map<BasicBlock*, Node*> Nodes;
  void reset();
  typedef std::map<BasicBlock*, Node*> NodeMapType;

  Node *RootNode;
public:
  class Node {
    friend class DominatorTree;
    friend class PostDominatorTree;
    friend class DominatorTreeBase;
    BasicBlock *TheBB;
    Node *IDom;
    std::vector<Node*> Children;
  public:
    typedef std::vector<Node*>::iterator iterator;
    typedef std::vector<Node*>::const_iterator const_iterator;

    iterator begin()             { return Children.begin(); }
    iterator end()               { return Children.end(); }
    const_iterator begin() const { return Children.begin(); }
    const_iterator end()   const { return Children.end(); }

    inline BasicBlock *getBlock() const { return TheBB; }
    inline Node *getIDom() const { return IDom; }
    inline const std::vector<Node*> &getChildren() const { return Children; }

    // dominates - Returns true iff this dominates N.  Note that this is not a
    // constant time operation!
    inline bool dominates(const Node *N) const {
      const Node *IDom;
      while ((IDom = N->getIDom()) != 0 && IDom != this)
	N = IDom;   // Walk up the tree
      return IDom != 0;
    }

  private:
    inline Node(BasicBlock *BB, Node *iDom)
      : TheBB(BB), IDom(iDom) {}
    inline Node *addChild(Node *C) { Children.push_back(C); return C; }

    void setIDom(Node *NewIDom);
  };

public:
  DominatorTreeBase(bool isPostDom) : DominatorBase(isPostDom) {}
  ~DominatorTreeBase() { reset(); }

  virtual void releaseMemory() { reset(); }

  /// getNode - return the (Post)DominatorTree node for the specified basic
  /// block.  This is the same as using operator[] on this class.
  ///
  inline Node *getNode(BasicBlock *BB) const {
    NodeMapType::const_iterator i = Nodes.find(BB);
    return (i != Nodes.end()) ? i->second : 0;
  }

  inline Node *operator[](BasicBlock *BB) const {
    return getNode(BB);
  }

  // getRootNode - This returns the entry node for the CFG of the function.  If
  // this tree represents the post-dominance relations for a function, however,
  // this root may be a node with the block == NULL.  This is the case when
  // there are multiple exit nodes from a particular function.  Consumers of
  // post-dominance information must be capable of dealing with this
  // possibility.
  //
  Node *getRootNode() { return RootNode; }
  const Node *getRootNode() const { return RootNode; }

  //===--------------------------------------------------------------------===//
  // API to update (Post)DominatorTree information based on modifications to
  // the CFG...

  /// createNewNode - Add a new node to the dominator tree information.  This
  /// creates a new node as a child of IDomNode, linking it into the children
  /// list of the immediate dominator.
  ///
  Node *createNewNode(BasicBlock *BB, Node *IDomNode) {
    assert(getNode(BB) == 0 && "Block already in dominator tree!");
    assert(IDomNode && "Not immediate dominator specified for block!");
    return Nodes[BB] = IDomNode->addChild(new Node(BB, IDomNode));
  }

  /// changeImmediateDominator - This method is used to update the dominator
  /// tree information when a node's immediate dominator changes.
  ///
  void changeImmediateDominator(Node *node, Node *NewIDom) {
    assert(node && NewIDom && "Cannot change null node pointers!");
    node->setIDom(NewIDom);
  }

  /// print - Convert to human readable form
  virtual void print(std::ostream &OS) const;
};


//===-------------------------------------
// DominatorTree Class - Concrete subclass of DominatorTreeBase that is used to
// compute a normal dominator tree.
//
struct DominatorTree : public DominatorTreeBase {
  DominatorTree() : DominatorTreeBase(false) {}

  BasicBlock *getRoot() const {
    assert(Roots.size() == 1 && "Should always have entry node!");
    return Roots[0];
  }

  virtual bool runOnFunction(Function &F) {
    reset();     // Reset from the last time we were run...
    DominatorSet &DS = getAnalysis<DominatorSet>();
    Roots = DS.getRoots();
    calculate(DS);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<DominatorSet>();
  }
private:
  void calculate(const DominatorSet &DS);
};

//===-------------------------------------
// DominatorTree GraphTraits specialization so the DominatorTree can be
// iterable by generic graph iterators.

template <> struct GraphTraits<DominatorTree::Node*> {
  typedef DominatorTree::Node NodeType;
  typedef NodeType::iterator  ChildIteratorType;

  static NodeType *getEntryNode(NodeType *N) {
    return N;
  }
  static inline ChildIteratorType child_begin(NodeType* N) {
    return N->begin();
  }
  static inline ChildIteratorType child_end(NodeType* N) {
    return N->end();
  }
};

template <> struct GraphTraits<DominatorTree*>
  : public GraphTraits<DominatorTree::Node*> {
  static NodeType *getEntryNode(DominatorTree *DT) {
    return DT->getRootNode();
  }
};

//===----------------------------------------------------------------------===//
//
// DominanceFrontier - Calculate the dominance frontiers for a function.
//
struct DominanceFrontierBase : public DominatorBase {
  typedef std::set<BasicBlock*>             DomSetType;    // Dom set for a bb
  typedef std::map<BasicBlock*, DomSetType> DomSetMapType; // Dom set map
protected:
  DomSetMapType Frontiers;
public:
  DominanceFrontierBase(bool isPostDom) : DominatorBase(isPostDom) {}

  virtual void releaseMemory() { Frontiers.clear(); }

  // Accessor interface:
  typedef DomSetMapType::iterator iterator;
  typedef DomSetMapType::const_iterator const_iterator;
  iterator       begin()       { return Frontiers.begin(); }
  const_iterator begin() const { return Frontiers.begin(); }
  iterator       end()         { return Frontiers.end(); }
  const_iterator end()   const { return Frontiers.end(); }
  iterator       find(BasicBlock *B)       { return Frontiers.find(B); }
  const_iterator find(BasicBlock *B) const { return Frontiers.find(B); }

  void addBasicBlock(BasicBlock *BB, const DomSetType &frontier) {
    assert(find(BB) == end() && "Block already in DominanceFrontier!");
    Frontiers.insert(std::make_pair(BB, frontier));
  }

  void addToFrontier(iterator I, BasicBlock *Node) {
    assert(I != end() && "BB is not in DominanceFrontier!");
    I->second.insert(Node);
  }

  void removeFromFrontier(iterator I, BasicBlock *Node) {
    assert(I != end() && "BB is not in DominanceFrontier!");
    assert(I->second.count(Node) && "Node is not in DominanceFrontier of BB");
    I->second.erase(Node);
  }

  // print - Convert to human readable form
  virtual void print(std::ostream &OS) const;
};


//===-------------------------------------
// DominatorTree Class - Concrete subclass of DominatorTreeBase that is used to
// compute a normal dominator tree.
//
struct DominanceFrontier : public DominanceFrontierBase {
  DominanceFrontier() : DominanceFrontierBase(false) {}

  BasicBlock *getRoot() const {
    assert(Roots.size() == 1 && "Should always have entry node!");
    return Roots[0];
  }

  virtual bool runOnFunction(Function &) {
    Frontiers.clear();
    DominatorTree &DT = getAnalysis<DominatorTree>();
    Roots = DT.getRoots();
    assert(Roots.size() == 1 && "Only one entry block for forward domfronts!");
    calculate(DT, DT[Roots[0]]);
    return false;
  }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<DominatorTree>();
  }
private:
  const DomSetType &calculate(const DominatorTree &DT,
                              const DominatorTree::Node *Node);
};

#endif

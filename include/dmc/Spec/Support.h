#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/SmallVector.h>

namespace dmc {

/// An immutable list that self-sorts on creation.
template <typename T>
struct ImmutableSortedList : public llvm::SmallVector<T, 4> {
  /// Sort on creation with comparator.
  template <typename Container, typename ComparatorT>
  ImmutableSortedList(const Container &c,
                      ComparatorT comparator = ComparatorT{})
      : llvm::SmallVector<T, 4>{std::begin(c), std::end(c)} {
    llvm::sort(std::begin(*this), std::end(*this), comparator);
  }

  /// Compare list sizes and contents.
  bool operator==(const ImmutableSortedList<T> &other) const {
    if (this->size() != other.size())
      return false;
    return std::equal(this->begin(), this->end(), other.begin());
  }

  /// Hash list values.
  llvm::hash_code hash() const {
    return llvm::hash_combine_range(this->begin(), this->end());
  }
};

template <typename ComparatorT, typename T>
ImmutableSortedList<T> getSortedListOf(llvm::ArrayRef<T> arr) {
  return {arr, ComparatorT{}};
}

/// Wrapper for kind switches with an Arg instance.
template <typename ActionT, typename ArgT>
struct KindActionWrapper {
  const ActionT &action;
  ArgT base;

  template <typename ConcreteType>
  auto operator()() const {
    return action(base.template cast<ConcreteType>());
  }
};

template <typename RetT, typename ParserT>
struct ParseAction {
  ParserT &parser;

  template <typename ConcreteType>
  RetT operator()() const {
    return ConcreteType::parse(parser);
  }
};

template <typename PrinterT>
struct PrintAction {
  PrinterT &printer;

  template <typename ConcreteType>
  int operator()(ConcreteType base) const {
    base.print(printer);
    return 0;
  }
};

template <typename ArgT>
struct VerifyAction {
  ArgT arg;

  template <typename ConcreteType>
  mlir::LogicalResult operator()(ConcreteType base) const {
    return base.verify(arg);
  }
};


} // end namespace dmc

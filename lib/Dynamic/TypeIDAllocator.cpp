#include "dmc/Dynamic/TypeIDAllocator.h"

using namespace mlir;

namespace dmc {

/// Pre-allocate a pool of TypeIDs. Definitely a hack.
namespace {

template <std::size_t NumIDs>
using IDList = std::array<TypeID, NumIDs>;

namespace detail {

template <std::size_t Index> struct IDReserve {};

template <std::size_t... ID>
IDList<sizeof...(ID)> allocateIDPoolImpl(std::index_sequence<ID...>) {
  return {TypeID::get<IDReserve<ID>>()...};
}

template <std::size_t NumIDs> auto allocateIDPool() {
  return allocateIDPoolImpl(std::make_index_sequence<NumIDs>());
}

} // end namespace detail

template <std::size_t NumIDs>
class FixedTypeIDAllocator : public TypeIDAllocator {
public:
  TypeID allocateID() override {
    assert(index < ids.size() && "Out of TypeIDs");
    return ids[index++];
  }

private:
  std::size_t index{};
  IDList<NumIDs> ids = detail::allocateIDPool<NumIDs>();
};

} // end anonymous namespace

TypeIDAllocator *getFixedTypeIDAllocator() {
  static FixedTypeIDAllocator<2048> typeIdAllocator;
  return &typeIdAllocator;
}

} // end namespace dmc

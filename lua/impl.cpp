#include "lua.h"

#include <iostream>
#include <string>
#include <unordered_map>
#include <cstddef>
#include <cassert>

extern "C" {

struct GCObject {
  int _ : 1;

  virtual ~GCObject() = default;
};

}

class LuaString;
class LuaTable;

class LuaObject {
public:
  LuaObject(TObject *impl) : impl{impl} {}

  /*****************************************************************************
   * Type handling
   ****************************************************************************/

  LuaType getType() const { return impl->type; }
  void setType(LuaType ty) { impl->type = ty; }

  template <LuaType ty> bool isa() const { return getType() == ty; }
  bool isNil() const { return isa<LuaType::Nil>(); }
  bool isBool() const { return isa<LuaType::Bool>(); }
  bool isNumber() const { return isa<LuaType::Number>(); }
  bool isString() const { return isa<LuaType::String>(); }
  bool isTable() const { return isa<LuaType::Table>(); }
  bool isFunction() const { return isa<LuaType::Function>(); }
  bool isUserData() const { return isa<LuaType::UserData>(); }
  bool isThread() const { return isa<LuaType::Thread>(); }

  /*****************************************************************************
   * Construction
   ****************************************************************************/

  void constructNil() {
    setType(LuaType::Nil);
  }
  void constructBool(bool b) {
    setType(LuaType::Bool);
    setBool(b);
  }
  void constructInt(LuaInteger iv) {
    setType(LuaType::Number);
    setNumberKind(Kind::Integer);
    setInt(iv);
  }
  void constructReal(LuaReal fp) {
    setType(LuaType::Number);
    setNumberKind(Kind::Real);
    setReal(fp);
  }

  void constructString(const char *data, int32_t len);
  void constructString(std::string val);
  void constructTable();

  /*****************************************************************************
   * Value handling
   ****************************************************************************/

  void setBool(bool b) {
    assert(isBool());
    impl->value.boolean = b;
  }
  bool getBool() const {
    assert(isBool());
    return impl->value.boolean;
  }

  void setNumberKind(Kind kind) {
    assert(isNumber());
    impl->value.number.kind = kind;
  }
  Kind getNumberKind() const {
    assert(isNumber());
    return impl->value.number.kind;
  }

  bool isInteger() const {
    return getNumberKind() == Kind::Integer;
  }
  bool isReal() const {
    return getNumberKind() == Kind::Real;
  }

  void setInt(LuaInteger iv) {
    assert(isInteger());
    impl->value.number.iv = iv;
  }
  void setReal(LuaReal fp) {
    assert(isReal());
    impl->value.number.fp = fp;
  }

  LuaInteger getInt() const {
    assert(isInteger());
    return impl->value.number.iv;
  }
  LuaReal getReal() const {
    assert(isReal());
    return impl->value.number.fp;
  }

  void setGcPtr(GCObject *gc) {
    assert(isString() || isTable());
    impl->value.gc = gc;
  }
  GCObject *getGcPtr() {
    assert(isString() || isTable());
    return impl->value.gc;
  }
  const GCObject *getGcPtr() const {
    assert(isString() || isTable());
    return impl->value.gc;
  }

  LuaString *asString();
  const LuaString *asString() const;
  LuaTable *asTable();

  /*****************************************************************************
   * Impl access
   ****************************************************************************/

  TObject *getImpl() { return impl; }

  void copyTo(TObject *o) {
    o->type = impl->type;
    o->value = impl->value;
  }

private:
  TObject *impl;
};

class LuaString : public GCObject {
public:
  explicit LuaString(std::string val) : impl{std::move(val)} {}

  const std::string &get() const { return impl; }

private:
  std::string impl;
};

template <> struct std::hash<LuaObject> {
  std::size_t operator()(const LuaObject &obj) const {
    switch (obj.getType()) {
    case LuaType::Nil:
      return std::hash<std::nullptr_t>{}(nullptr);
    case LuaType::Bool:
      return std::hash<bool>{}(obj.getBool());
    case LuaType::Number:
      if (obj.isInteger()) {
        return std::hash<LuaInteger>{}(obj.getInt());
      } else {
        return std::hash<LuaReal>{}(obj.getReal());
      }
    case LuaType::String:
      return std::hash<std::string>{}(obj.asString()->get());
    default:
      throw std::invalid_argument{"Hash not implemented for this type"};
    }
  }
};

template <> struct std::equal_to<LuaObject> {
  bool operator()(const LuaObject &lhs, const LuaObject &rhs) const {
    if (lhs.getType() != rhs.getType()) {
      return false;
    }
    switch (lhs.getType()) {
    case LuaType::Nil:
      return true;
    case LuaType::Bool:
      return lhs.getBool() == rhs.getBool();
    case LuaType::Number:
      if (lhs.isInteger()) {
        if (rhs.isInteger()) {
          return lhs.getInt() == rhs.getInt();
        } else {
          return lhs.getInt() == rhs.getReal();
        }
      } else {
        if (rhs.isInteger()) {
          return lhs.getReal() == rhs.getInt();
        } else {
          return lhs.getReal() == rhs.getReal();
        }
      }
    case LuaType::String:
      return lhs.asString()->get() == rhs.asString()->get();
    default:
      throw std::invalid_argument{"Hash not implemented for this type"};
    }
  }
};

class LuaTable : public GCObject {
  using Table = std::unordered_map<LuaObject, LuaObject>;

public:
  void assign(LuaObject key, LuaObject val) {
    impl.insert_or_assign(key, val);
  }
  auto find(LuaObject key) {
    return impl.find(key);
  }

  auto begin() { return impl.begin(); }
  auto end() { return impl.end(); }
  auto size() const { return impl.size(); }

private:
  Table impl;
};

void LuaObject::constructString(const char *data, int32_t len) {
  setType(LuaType::String);
  std::string val(data, len);
  setGcPtr(new LuaString{std::move(val)});
}
void LuaObject::constructString(std::string val) {
  setType(LuaType::String);
  setGcPtr(new LuaString{std::move(val)});
}

void LuaObject::constructTable() {
  setType(LuaType::Table);
  setGcPtr(new LuaTable{});
}

LuaString *LuaObject::asString() {
  assert(isString());
  auto *ret = dynamic_cast<LuaString *>(getGcPtr());
  assert(ret);
  return ret;
}
const LuaString *LuaObject::asString() const {
  assert(isString());
  auto *ret = dynamic_cast<const LuaString *>(getGcPtr());
  assert(ret);
  return ret;
}

LuaTable *LuaObject::asTable() {
  assert(isTable());
  auto *ret = dynamic_cast<LuaTable *>(getGcPtr());
  assert(ret);
  return ret;
}

extern "C" {

void lua_add(TObject *sret, TObject *lhs, TObject *rhs) {
  LuaObject olhs = lhs, orhs = rhs, ret = sret;
  if (!olhs.isNumber()) {
    throw std::invalid_argument{"Left hand argument is not a number"};
  } else if (!orhs.isNumber()) {
    throw std::invalid_argument{"Right hand argument is not a number"};
  }
  if (olhs.isInteger()) {
    if (orhs.isInteger()) {
      ret.constructInt(olhs.getInt() + orhs.getInt());
    } else {
      ret.constructReal(olhs.getInt() + orhs.getReal());
    }
  } else {
    if (orhs.isInteger()) {
      ret.constructReal(olhs.getReal() + orhs.getInt());
    } else {
      ret.constructReal(olhs.getReal() + orhs.getReal());
    }
  }
}

void lua_sub(TObject *sret, TObject *lhs, TObject *rhs) {
  throw std::invalid_argument{"not implemented"};
}

void lua_eq(TObject *sret, TObject *lhs, TObject *rhs) {
  LuaObject obj = sret;
  obj.constructBool(std::equal_to<LuaObject>{}(lhs, rhs));
}

void lua_neq(TObject *sret, TObject *lhs, TObject *rhs) {
  throw std::invalid_argument{"not implemented"};
}


void lua_get_nil(TObject *sret) {
  LuaObject obj = sret;
  obj.constructNil();
}

void lua_new_table(TObject *sret) {
  LuaObject obj = sret;
  obj.constructTable();
}

void lua_get_string(TObject *sret, const char *data, int32_t len) {
  LuaObject obj = sret;
  obj.constructString(data, len);
}


void lua_wrap_int(TObject *sret, LuaInteger val) {
  LuaObject obj = sret;
  obj.constructInt(val);
}

void lua_wrap_real(TObject *sret, LuaReal val) {
  LuaObject obj = sret;
  obj.constructReal(val);
}

void lua_wrap_bool(TObject *sret, bool boolean) {
  LuaObject obj = sret;
  obj.constructBool(boolean);
}

LuaInteger lua_unwrap_int(TObject *val) {
  LuaObject obj = val;
  if (!obj.isNumber()) {
    throw std::invalid_argument{"Lua value is not a number"};
  } else if (!obj.isInteger()) {
    throw std::invalid_argument{"Lua value is not an integer"};
  }
  return obj.getInt();
}

LuaReal lua_unwrap_real(TObject *val) {
  LuaObject obj = val;
  if (!obj.isNumber()) {
    throw std::invalid_argument{"Lua object is not a number"};
  } else if (!obj.isReal()) {
    throw std::invalid_argument{"Lua number is not a real"};
  }
  return obj.getReal();
}

bool lua_unwrap_bool(TObject *val) {
  LuaObject obj = val;
  if (!obj.isBool()) {
    throw std::invalid_argument{"Lua object is not a boolean"};
  }
  return obj.getBool();
}


void lua_typeof(TObject *sret, TObject *val) {
  LuaObject obj = val, ret = sret;
  switch (obj.getType()) {
  case LuaType::Nil:
    ret.constructString("nil");
    break;
  case LuaType::Bool:
    ret.constructString("boolean");
    break;
  case LuaType::Number:
    ret.constructString("number");
    break;
  case LuaType::String:
    ret.constructString("string");
    break;
  case LuaType::Table:
    ret.constructString("table");
    break;
  default:
    throw std::invalid_argument{"unsupported typeof type"};
  }
}


void lua_table_get(TObject *sret, TObject *tbl, TObject *key) {
  LuaObject obj = tbl;
  if (!obj.isTable()) {
    throw std::invalid_argument{"Indexed Lua object is not a table"};
  }
  auto *p = obj.asTable();
  auto it = p->find(key);
  if (it == p->end()) {
    lua_get_nil(sret);
  } else {
    it->second.copyTo(sret);
  }
}

void lua_table_set(TObject *tbl, TObject *key, TObject *val) {
  LuaObject obj = tbl;
  if (!obj.isTable()) {
    throw std::invalid_argument{"Indexed Lua object is not a table"};
  }
  obj.asTable()->assign(key, val);
}

void lua_table_size(TObject *sret, TObject *tbl) {
  LuaObject obj = tbl;
  if (!obj.isTable()) {
    throw std::invalid_argument{"Lua object is not a table"};
  }
  LuaObject ret = sret;
  ret.constructInt(obj.asTable()->size());
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

void print(TObject *val) {
  using namespace std;
  LuaObject obj = val;

  switch (obj.getType()) {
  case LuaType::Nil:
    cout << "nil";
    break;
  case LuaType::Bool:
    cout << (obj.getBool() ? "true" : "false");
    break;
  case LuaType::Number:
    if (obj.isInteger()) {
      cout << obj.getInt();
    } else {
      cout << obj.getReal();
    }
    break;
  case LuaType::String:
    cout << obj.asString()->get();
    break;
  case LuaType::Table:
    cout << "table: " << reinterpret_cast<void *>(obj.asTable());
    break;
  default:
    cout << "unknown";
    break;
  }
  cout << endl;
}

void random_string_or_int(TObject *sret, size_t len) {
  lua_wrap_int(sret, 4);
}

}

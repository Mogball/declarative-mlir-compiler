#pragma once

namespace dmc {
class DynamicDialect;
namespace py {
void exposeDialectInternal(DynamicDialect *dialect);
} // end namespace py
} // end namespace dmc

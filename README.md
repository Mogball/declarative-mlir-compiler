# Declarative MLIR Compilers

[Design document](https://docs.google.com/document/d/1eAgIQZZ2dItJFSrCxemt7fwH0CD4w6_ueLKVl6UL-NU/edit?usp=sharing)

## Build Requirements

- `cmake>=3.10`

Build with `cmake --build . --target DMC<lib>`.

## Short-Term TODOs

- Dynamic attribute definitions

## Long-Term TODOs

- Incremental parsing: dynamic objects are registered during the parse, which
  means that references to dynamic objects (e.g. types and attributes) can be
  resolved on-the-go.
- Add graceful failures (e.g. multiple Ops with the same name) instead of
  relying on MLIR asserts to better support dynamic environment.
- Integrate `Location` propagation for all API layer.

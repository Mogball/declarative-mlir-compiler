# Declarative MLIR Compilers

[Design document](https://docs.google.com/document/d/1eAgIQZZ2dItJFSrCxemt7fwH0CD4w6_ueLKVl6UL-NU/edit?usp=sharing)

## Build Requirements

- `cmake >= 3.10`
- `python >= 3.6`
- `boost >= 1.7`

Build with `cmake --build . --target gen`.

### Arch Linux

```
sudo pacman -Sy cmake python boost
```

## Short-Term TODOs

- Region and successor specifiers

## Long-Term TODOs

- Incremental parsing: dynamic objects are registered during the parse, which
  means that references to dynamic objects (e.g. types and attributes) can be
  resolved on-the-go.

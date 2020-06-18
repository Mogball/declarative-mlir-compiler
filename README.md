# Declarative MLIR Compilers

[Design document](https://docs.google.com/document/d/1eAgIQZZ2dItJFSrCxemt7fwH0CD4w6_ueLKVl6UL-NU/edit?usp=sharing)

## Build Requirements

- `cmake >= 3.10`
- `python >= 3.6`
- `antlr >= 4`

Build with `cmake --build . --target gen`.

### Arch Linux

```
sudo pacman -Sy cmake python antlr4
```

### MacOS

```
brew install cmake python3
```

### Python Dependencies

The ANTLR4 runtime is needed.

```
pip3 install antlr4-python3-runtime
```

## Short-Term TODOs

- Automatic bindings for custom ops and types

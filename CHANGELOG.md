## 0.4.0 (2023-12-27)

### Feat

- implement horizontal `vvg`
- improve projection performance

## 0.3.0 (2023-12-26)

### Feat

- implement basic vvg generation from `ts2vg`
- **nvg**: add divide-and-conquer (dc) version

### Fix

- **nvg**: parallelize with `numba`
- **dc**: remove `divide-and-conquer` version

### Refactor

- export input ensurance to own function
- export visibility check to own function

## 0.2.0 (2023-12-24)

### Feat

- convert from `torch` to `numpy/numba`

### Fix

- remove `@torch.jit.script` for compatibility
- update `unweighted` to `weighted`

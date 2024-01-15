## 0.7.1 (2024-01-15)

### Refactor

- format code

## 0.7.0 (2024-01-01)

### Feat

- allow import from package root

### Fix

- **vvg**: ensure input dimension check is correct
- add `unique` option to `WeightMethod` enum

### Refactor

- alphabetically sort `WeightMethod` enums

## 0.6.1 (2023-12-29)

### Fix

- in attempt to avoid flaky tests remove cache

## 0.6.0 (2023-12-28)

### Feat

- **vvg**: make `undirected` default like `ts2vg`
- **vvg**: add custom weight calculation

### Fix

- normalize projections to have correct weights

### Refactor

- **ts2vg**: export naive `ts2vg` sanity check
- simplify code structure and rename

## 0.5.0 (2023-12-27)

### Feat

- **lpvg**: add penetrable limit to visibility

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

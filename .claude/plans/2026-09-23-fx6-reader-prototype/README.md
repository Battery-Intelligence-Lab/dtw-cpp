# FX-6 text-reader prototype (2026-09-23)

Written by the reader audit of 2026-09-23 on a scratch copy of `dtwc/`, against `e784e5c`; never applied
to the tree. `reader.diff` (+117 / −94 in `fileOperations.hpp` and `DataLoader.hpp`) routes the float parse
through fast_float, checks the BOM by `peek`, trims ASCII-only, replaces four line loops with
`for_each_data_line` (trailing blank lines ignored, a blank line followed by data is an error), requires one
value per line in folder files, and skips dot-files. It expects fast_float at `dtwc/extern/fast_float/` —
PLAN FX-6 moves the parse into one `.cpp` helper instead, so the header stays out of installed headers.
`cases/` and `folder_cases/` are the input matrix the audit ran; turn them into tests, then delete this folder.

---
name: Bug report
about: Something crashes, raises unexpectedly, or behaves differently from its documentation
title: "[bug] "
labels: bug
---

<!-- For security problems, do NOT use this form: see SECURITY.md. -->
<!-- For a numerical result that disagrees with a reference, use the
     "Validation discrepancy" template instead. -->

## What happened

A clear description of the problem and what you expected instead.

## Minimal reproduction

```python
import photonix as px
# the smallest snippet that shows the problem
```

Full traceback (if any):

```text
paste here
```

## Environment (required)

Paste the output of `photonix info --json` (or `python -m photonix info --json`):

```json
paste here
```

Was `PHOTONIX_BACKEND=numpy` set?  yes / no

## Anything else

Workarounds you found, related issues, a suspected cause.

# Overview

Parallel Linear Algebra (parallela) is a C++ library for linear algebra on hybrid distributed and shared memory systems.

Goals:
- Be modular
- Keep lowest level functionality as general as possible
- Use BLAS/MKL whenever possible
- Add features/tests/examples concurrently
- Use clang-format
- Use vscode plugin for deoxygen compatible documentation 

---
## Structure

- Level 1: Basic operations and memory management
- Level 2: Basic Linear Algebra Routines (e.g. Gram-Schmidt)
- Level 3: More Advanced Linear Algebra Routines (e.g. Davidson)
- Level 4: Python Wrappers for selected routines

---
## Dependencies
- Eigen3
- Catch2
- MKL (optional)
  
---
## Notes

### Logging
Logging and all important output should be dumbed to the appropriate json file.
# Coding conventions

- Avoid using "using" in the global namespace.
- Include project-specific headers before standard headers.
- Use descriptive names for variables.
- Break down complex conditionals into multiple statements.
- Minimise nested statements.

### Code editor

We recommend using [Visual Studio Code (VS Code)](https://code.visualstudio.com/). Enable `format on save` to automatically enforce `.clang-format`.

## Copyright
- Verify that contributed code and dependencies respect all license terms.
- Our software uses a permissive BSD-3-Clause license; avoid incorporating copyleft libraries.
- Report any copyright violations by opening an issue immediately. We will remove any infringing content as soon as possible.
- Using AI tools to help write code is acceptable as long as it is thoroughly reviewed by a human developer and does not infringe on copyrights.

## Versioning

We follow semantic versioning (`MAJOR.MINOR.PATCH`):

- `PATCH`: Backwards-compatible bug fixes.
- `MINOR`: Backwards-compatible new features.
- `MAJOR`: Incompatible API or behaviour changes.

## Issue management

As a small library developed by a small team, we review issues informally. To avoid duplicate efforts, please comment on the issue you are working on.

## Commenting

Use Doxygen-compatible comments, and feel free to update any incompatible comments you encounter. Refer to the [Doxygen guidelines](https://doxygen.nl/manual/docblocks.html) for commenting best practices.

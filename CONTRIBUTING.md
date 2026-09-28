# Contributing to OsteotomyCuts

Thank you for helping to improve OsteotomyCuts. Bug reports, clinical feedback and code are all welcome.

## Before you start

- **Contributor Licence Agreement.** Every contributor must sign the [Individual Contributor Licence Agreement](CLA.md) before a pull request can be merged. The CLA Assistant bot asks you to sign when you open your first pull request. The CLA lets the project stay available under GPL-3.0-or-later and also be offered under a commercial licence.
- **No patient data.** Never attach DICOM files, patient models, screenshots with patient details or any other personal data to issues or pull requests. Use synthetic geometry or de-identified models only.
- **Not a medical device.** Read the [disclaimer](DISCLAIMER.md).

## Reporting a problem

Open an issue with:

1. the Slicer version and operating system;
2. what you did (steps, settings, the kind of model and cut);
3. what you expected and what happened, with the Python console output if there was an error.

## Code

- Follow the structure of the module: Module / Widget / Logic / Test classes. All computation is in the Logic class and must run without the GUI; module state lives in the parameter node; user-interface changes go in `Resources/UI/*.ui`.
- The core cut uses implicit (signed) distances to the cutting surface and clipping, not mesh booleans.
- Style: PEP 8, type hints, docstrings on all Logic methods, British English in user-facing text and comments. User-facing labels and messages use surgical words; technical terms belong in tooltips.
- Every source file starts with the SPDX header:

  ```
  # SPDX-FileCopyrightText: <year> <your name>
  # SPDX-License-Identifier: GPL-3.0-or-later
  ```

- Add or update tests (synthetic geometry) and run them headlessly before opening a pull request:

  ```
  Slicer --no-splash --no-main-window --python-script OsteotomyCuts/Testing/Python/run_headless_tests.py
  ```

  The exit code is 0 when all tests pass.
- Keep commits small and focused, with clear messages.

## Licence

By contributing you agree that your contributions are licensed as described in the [CLA](CLA.md); the project is distributed under the [GNU General Public License, version 3 or later](LICENSE).

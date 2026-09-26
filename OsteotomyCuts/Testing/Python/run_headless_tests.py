"""Headless test runner for the Osteotomy Cuts module.

Run with Slicer's --python-script option. Always exits: 0 when all tests pass, 1 otherwise.
slicer.util.selectModule() is not used because it needs a main window.
"""

import traceback

import slicer

exitCode = 1
try:
    # Builds the module widget (loads the .ui) without needing a main window
    widget = slicer.modules.osteotomycuts.widgetRepresentation()
    assert widget is not None, "module widget could not be created"
    import OsteotomyCuts
    OsteotomyCuts.OsteotomyCutsTest().runTest()
    print("ALL TESTS PASSED")
    exitCode = 0
except Exception:
    traceback.print_exc()
finally:
    slicer.util.exit(exitCode)

"""MAAA — minimal reference core.

The smallest complete statement of the five-layer loop: one class per layer,
no I/O, no threads, no database. ``layers/`` is the reference runtime; this
package is the readable version of the same idea and the one the tests in
tests/test_maaa_core.py and examples/run_demo.py exercise.

Both share their risk bands, brevity limit and failsafe thresholds through
maaa_config (config/default.yaml) and report the same version.
"""

from maaa_config import VERSION as __version__   # noqa: F401  (re-exported)

"""Pin BLAS to one thread so results are bit-reproducible.

Multithreaded BLAS chooses its reduction order dynamically, so a dot product can
sum its terms in a different sequence on two runs of identical code. Floating
point addition is not associative, and the resulting disagreement is ~1e-16 per
operation. That is invisible almost everywhere and decisive here.

How it surfaced in this study
-----------------------------
The HMM detector runs up to 200 EM iterations per refit, each one a chain of BLAS
reductions, so the ~1e-16 disagreement compounds into a posterior that differs at
~1e-14 between runs. Whether that matters depends entirely on how a mechanism
*consumes* the posterior:

* `regime_hmm` takes ``argmax`` and looks up a parameter block. A 1e-14 wobble
  essentially never flips the argmax, so the arm reproduced bit-identically and
  the harness's determinism check passed.
* `tilt_hmm` weights its tilt by the posterior *continuously*. The same wobble
  passes straight into ``mu``, moves the optimum, and the swarm amplifies it into
  weight differences of ~5e-14 across ~2500 cells.

So the determinism guarantee had been holding only because every mechanism tested
until then quantized the detector's output. That is worth stating plainly: a
reproducibility check that passes can be passing for the wrong reason, and the
only way to find out is to re-execute and diff the decisions
(:mod:`fpso.experiments.verify_reproduction`).

The trade
---------
Single-threaded BLAS is slower. For this workload the cost is small — the panel
is 50 assets wide and the heavy loop is the swarm, which is not BLAS-bound — and
a study whose subject is measurement error does not get to trade reproducibility
for speed. Set ``FPSO_ALLOW_THREADED_BLAS=1`` to opt out for exploratory work;
never set it for a run whose numbers will be reported.

The variables must be set before the BLAS library is loaded, which is why this
module is imported at the top of :mod:`fpso` and imports nothing itself.
"""

from __future__ import annotations

import os

__all__ = ["THREAD_VARIABLES", "OPT_OUT_VARIABLE", "pin_blas_threads", "threading_report"]

THREAD_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate, which threadpoolctl cannot see
    "NUMEXPR_NUM_THREADS",
)
"""Every backend that could reorder a reduction on the platforms this runs on.

All five are set rather than the one believed to be active: which BLAS numpy
binds to varies by wheel and by machine, and a missed variable produces exactly
the intermittent, hard-to-attribute drift this module exists to remove.
"""

OPT_OUT_VARIABLE = "FPSO_ALLOW_THREADED_BLAS"


def pin_blas_threads() -> bool:
    """Set the thread-count variables to 1 unless told otherwise.

    An existing value is never overwritten: a caller who has deliberately set
    ``OMP_NUM_THREADS`` knows something this module does not, and silently
    overriding them would be its own reproducibility problem.

    Returns:
        True if single-threaded execution is now requested, False if the caller
        opted out.
    """
    if os.environ.get(OPT_OUT_VARIABLE) == "1":
        return False
    for name in THREAD_VARIABLES:
        os.environ.setdefault(name, "1")
    return True


def threading_report() -> dict[str, str]:
    """Current values, for recording in a manifest or printing in a banner.

    Worth persisting alongside results: a run that silently used eight threads is
    not comparable at the bit level with one that used one, and without this the
    difference is undetectable after the fact.
    """
    report = {name: os.environ.get(name, "<unset>") for name in THREAD_VARIABLES}
    report[OPT_OUT_VARIABLE] = os.environ.get(OPT_OUT_VARIABLE, "0")
    return report

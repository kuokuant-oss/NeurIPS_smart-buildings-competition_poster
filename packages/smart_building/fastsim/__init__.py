"""Opt-in fast building simulator for local training and validation (docs/plan/fastsim.md).

Off unless asked for. install("rom") or install("exact") patches the competition simulator of this process only
(patch.py); SB_FASTSIM=rom|exact asks for it through the environment, for code that calls install_from_env() and
for every process started by `scripts/fastsim.py run` (a sitecustomize hook installs it at interpreter start).
Policies and the submission path (scripts/build_submission.py, scripts/test_notebook.py, the Kaggle notebook)
never import this package, so a submission always runs on the official simulator.

Modes:
  exact  the official discretisation and field; one triangular solve per step instead of two, cached zone means
         and reward info, the convection shuffle in one compiled call. Same results up to rounding.
  rom    a reduced thermal model of the same discretisation (rom.py), built once per scenario and cached; the
         convection shuffle is left out. Zone temperatures within about 1e-4 K of the full field.
"""
import os

ENV = "SB_FASTSIM"
MODES = ("exact", "rom")


def install(mode="rom", **rom_options):
    from smart_building.fastsim import patch
    patch.install(mode, rom_options)
    return mode


def uninstall():
    from smart_building.fastsim import patch
    patch.uninstall()


def installed():
    from smart_building.fastsim import patch
    return patch.installed()


def requested(environ=None):
    """The mode SB_FASTSIM asks for, or None (unset, empty, 0, off or official)."""
    value = (environ if environ is not None else os.environ).get(ENV, "").strip().lower()
    if value in ("", "0", "off", "official", "none"):
        return None
    if value not in MODES:
        raise ValueError(f"{ENV}={value!r}: expected one of {MODES} or off")
    return value


def install_from_env():
    mode = requested()
    return install(mode) if mode else None

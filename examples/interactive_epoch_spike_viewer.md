# Interactive Epoch Spike Viewer

Minimum steps to run the example, from a Windows command prompt in the
checked-out `ipfx` repo root:

```
pip install -e .
pip install PySide6
python examples\interactive_epoch_spike_viewer.py
```

- `pip install -e .` installs `ipfx` (and its existing dependencies, incl.
  matplotlib) from this checked-out repo.
- `PySide6` is not an `ipfx` dependency and must be installed separately --
  it provides the GUI controls (the plot itself is matplotlib).
- An NWB file path can optionally be passed on the command line to load it
  on startup, e.g.:
  ```
  python examples\interactive_epoch_spike_viewer.py path\to\file.nwb
  ```
  Otherwise, use the "Open NWB File..." button or paste a path into the
  text box once the window is open.

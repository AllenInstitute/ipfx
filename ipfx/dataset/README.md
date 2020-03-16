README
======

`EphysDataSet` acts as the "one-stop" interface for accessing intracellular electrophysiology data. In particular, an object implementing the `EphysDataSet` interface provides all of the information required to run sweep extraction, auto qc, and feature extraction. While this information is likely stored in some external file or database (such as an [NWB2 file](https://pynwb.readthedocs.io/en/stable/) `EphysDataSet` does not directly communicate with these external stores. Rather, `EphysDataSet` relies on a specialized `EphysDataInterface` to load and organize the data, then presents a consistent set of attributes and methods for accessing it. This separation means that code can use an `EphysDataSet` without worrying about how the data is ultimately stored under the hood.


Changes from version 0.1.0 (to version 1.0.0)
=============================================


Changes from commit 1f2a56 (internal interest only, delete this!)
=================================================================

- `EphysDataSet`'s constructor no longer accepts sweep_info. This should instead be provided by the `EphysDataInterface`.
- `EphysDataSet` still exposes an `ontology` attribute, but this is a property which returns the ontology of it's `EphysDataInterface`. Rationale: we want these to be consistently identified.
- `extract_sweep_info` and `build_sweep_table` are gone. These methods were only called internally by `EphysDataSet`, not accessed publicly. Instead, we should provide a `sweep_table` property.
- `voltage_current` is no longer an instance method of `EphysDataSet`, since it accesses no attributes of `EphysDataSet` instances. It is instead a classmethod. I'm frankly skeptical of even that.
- `sweep` does *way less stuff*. The `EphysDataSet` should absolutely not be rescaling data. How does it know how the data are scaled in the source?

also, we're restoring some accessors. We want these implemented on the `EphysDataSet` so that people don't go reaching into the NWB interface. It also helps with backwards compatibility:
- get_clamp_mode
- get_stimulus_code
- get_stimulus_code_ext
- get_stimulus_name (the bulk of this should be moved to the stimulus ontology)
- get_recording_date
- get_stimulus_units

We are also making accessors for a bunch of data currently only on the `EphysDataInterface`. This is for the same reason as the above (sans backwards compatibility):
- acquisition_start_time
- session_start_time
- recording_date (or just a datetime for the above)
- get_real_sweep_number
- 
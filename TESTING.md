# Testing

The tests in the `tests` folder use pytest and can be run locally as:

```
SKIP_LIMS=true TEST_INHOUSE=false pytest
```

This does not assume that you have access to AllenInstitute internal resources like LIMS or network shares.

## Strategy for adopting newer NWB schema versions

The upstream source of NWB files for ipfx is [MIES](https://github.com/AllenInstitute/MIES) and
[IPNWB](https://github.com/AllenInstitute/IPNWB). We are at the moment using fixed versions of nwb-schema for
IPNWB and pynwb for ipfx. This is done in order to reduce maintenance and friction between the two.

Steps for upgrading:

- Choose a pynwb/nwb-schema version you want to adopt
- Raise pynwb version in `requirements.txt`
- Check if all tests still pass. In the past this always required patching the latest
  pynwb version, bringing that patch upstream and waiting for a new release.
- Fix pynwb deprecations where appropriate
- Upgrade the nwb-schema version used in IPNWB as documented there
- Re-export the files `Vip-IRES-Cre;Ai14(IVSCC)-226110.03.01.pxp` and
  `Vip-IRES-Cre;Ai14(IVSCC)-236654.04.02.pxp` using the IPNWB/MIES version with the new nwb-schema to NWBv2
- Add the files to `tests/data` and extend `testdata` in `tests/test_run_feature_vector.py` with them
- Check if the tests still pass
- Propose a PR

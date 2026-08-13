# TLSimu GenTL Producer simulator

Third-party binaries used by the test suite to exercise the `harvesters` camera interface without GenICam hardware.
Nothing in `src/` depends on these files, and they are excluded from the installable wheel. The source distribution
carries them alongside the test suite, so that a released version stays reproducible.

`TLSimu.cti` is the GenTL Producer simulator from the GenICam reference implementation. It presents four simulated
devices (`TLSimuMono` and `TLSimuColor`, two of each across two interfaces) and serves synthetic frames through the
regular GenTL buffer flow, so `HarvestersCamera` drives it through the same code path it uses for a real camera.
`libVirtualFG` is the virtual frame grabber that `TLSimu.cti` links against, and it must stay in the same directory.
Each producer resolves it relative to its own location, through `$ORIGIN` on Linux, `@loader_path` on macOS, and the
loader's DLL search order on Windows.

## Provenance

The binaries were extracted from the `genicam` 1.5.0 wheels bundled in `GenICam_Package_2025.10.zip`, published by the
EMVA at <https://www.emva.org/standards-technology/genicam/genicam-downloads/>. The wheels published to PyPI under the
same name omit the simulator, so the EMVA package is the only source.

| File                                | SHA-256                                                            |
|-------------------------------------|--------------------------------------------------------------------|
| `linux_x86_64/TLSimu.cti`           | `3efbeb7c4c7eebb6f6da2ceb9ea8db9d4c8a0062e20b211258ef542d7b9a7ce1` |
| `linux_x86_64/libVirtualFG.so`      | `aac43a4c4b5a352fe2170bf048ea398f617957453f1726d83d2a39d0bbfb0052` |
| `macos_universal2/TLSimu.cti`       | `dc75c072122bc3e4f0498675aa8bdc33a162c660fde4edcd8467b98bcf197910` |
| `macos_universal2/libVirtualFG.dylib` | `7df029779f22b76f9331aec04feecefbd2587f0c7d81287a37a8851053e40140` |
| `windows_amd64/TLSimu.cti`          | `62fe5aa2750d1e4d5b0fb4a7b97afbbd28ee6d1f75807855bc4932d93dbcc485` |
| `windows_amd64/VirtualFG.dll`       | `a822df43f60a9dd9751620b5ce9647aa6fed72c254721f836df27b2937503c49` |

The macOS binaries are universal (`x86_64` and `arm64`). The EMVA package ships no Linux `aarch64` build, so the
simulator-backed tests skip on that platform. On macOS they run on Apple Silicon under Python 3.12 and 3.13, which is
where the library installs a GenICam runtime to load a Producer with, and skip on the Intel and Python 3.14 hosts that
install none.

To refresh, download the current `GenICam_Package_<version>.zip`, unpack the `PythonWheels` archive for each platform
under `Reference Implementation/`, and copy `TLSimu.cti` together with its `VirtualFG` companion out of any wheel in the
resulting `wheelhouse` directory.

## License

These binaries are redistributed under the EMVA GenICam license reproduced in `LICENSE.txt`, which is the license the
EMVA ships with the `genicam` wheels. It applies to the contents of this directory only.

The text is BSD-style and carries the three standard clauses, but its grant reads "Redistribution and use in source
and binary forms, without modification, are permitted" rather than the "with or without modification" of the canonical
BSD 3-Clause license. Redistributing these binaries unmodified, as this directory does, is therefore permitted, while
modifying them is not.

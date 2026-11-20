# Download the example fields

The current installation contains twelve inputs: the nine fields described below plus three [CCC screenshot intensity fields](DATA_SOURCES.md). The two globe views are not asserted to be independent simulation datasets. Their preparation uses explicit background repair and square padding before reduction; the no-padding/no-cropping description below applies to the downloaded arrays.

From the repository root, run:

```sh
uv run --locked --extra datasets python -m task_app.dataset_download
```

This installs the optional HDF5 reader, downloads one scalar image or snapshot per source, and writes prepared NPY arrays and provenance JSON to `task_app/datasets/`. Restart Scalar Task Studio to see them in the dataset column. Open `task_app/datasets/preview.png` to compare their appearance. Re-running reuses valid prepared files.

The downloader targets seven additions: five different simulation problems and two biomedical intensity images. With the existing terrain there are eight scientific examples; the two synthetic controls bring the picker to ten options. These additions come from The Well and scikit-image, broadening the search beyond the original [SciVis catalog](https://github.com/huyen-nguyen/scivis-catalog). They are not ten slices or time steps from one dataset, and are not a popularity ranking. The six successfully downloaded fields have been visually reviewed in the preview: nuclei, vessels, maze patterns, smooth bands, shocks, and waves provide distinct structures.

| ID | Source and chosen scalar | Native size | Expected appearance |
| --- | --- | --- | --- |
| `mitosis` | [Human cell nuclei](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.human_mitosis): fluorescence intensity | 512 × 512 | Bright localized nuclei |
| `retinal_lesions` | [Retinal microaneurysms](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.microaneurysms): published green-channel crop | 102 × 102 | Dark lesions and vessels |
| `gray_scott` | [Gray–Scott reaction–diffusion](https://polymathic-ai.org/the_well/datasets/gray_scott_reaction_diffusion/): species B concentration | 128 × 128 | Spots and winding patterns |
| `active_matter` | [Active matter](https://polymathic-ai.org/the_well/datasets/active_matter/): concentration | 256 × 256 | Interacting bands and variations |
| `euler_shocks` | [Euler gas dynamics](https://polymathic-ai.org/the_well/datasets/euler_multi_quadrants_openBC/): density | 512 × 512 | Shock fronts and interfaces |
| `acoustic_waves` | [Acoustic scattering](https://polymathic-ai.org/the_well/datasets/acoustic_scattering_discontinuous/): pressure | 256 × 256 | Propagating and reflected waves |
| `viscoelastic_flow` | [Viscoelastic channel flow, corrected v2](https://polymathic-ai.org/the_well/datasets/viscoelastic_instability_v2/): pressure | 512 × 512 | Elongated channel structures |

The biomedical values are stored image intensities, not calibrated concentrations or clinical measurements. Their published images are CC0. The Well datasets are CC-BY-4.0; the manifest includes individual source citations. The deprecated viscoelastic v1 is not used.

## Small transfers

The two images have a 2 MiB cap each. The simulation collections contain large HDF5 files, but the script uses HTTP byte ranges to read only one scalar snapshot plus necessary file metadata. Each simulation has a 10 MiB payload cap. A server ignoring byte ranges is rejected; the script never falls back to downloading the complete container. It stops starting new range requests after two minutes or 160 requests for a source, with a 20-second timeout per request.

The default **64 MiB total payload cap** includes source discovery metadata. Python package installation and HTTP headers are separate. Large HDF5 chunks, unavailable hosts, or changed source layouts can prevent an individual source from fitting the cap. That source is skipped with an explanation; successful fields remain available. The run reports actual transferred bytes and writes `download-report.json`. Exit code 1 means at least one source failed. Re-run to retry missing sources.

The user-run download on 2026-11-06 successfully prepared six sources: mitosis, retinal lesions, Gray–Scott, active matter, Euler shocks, and acoustic waves. Total payload was 2,577,469 bytes (2.46 MiB). The viscoelastic v2 endpoint returned HTTP 401 and was skipped; no authentication requirement or alternative access has been established. All six files passed app loading and generated a task with image, CSV, Markdown, ZIP, and source provenance. The resulting picker has nine options: seven scientific examples including terrain, plus two controls. The transport limits also have local fixture tests. The optional `h5py` dependency is recorded in `pyproject.toml`, and the updated `uv.lock` passed `uv lock --check --offline`.

For only the two small images:

```sh
uv run --locked --extra datasets python -m task_app.dataset_download --only mitosis retinal_lesions
```

Use `--list` to inspect the selection without contacting data hosts, `--only gray_scott euler_shocks` to choose sources, or `--max-side 64` to prepare smaller arrays. `--max-total-mib` sets the total cap, from 1 to 256 MiB. A custom `--output` directory is supported, but only the default directory is loaded automatically by the app.

## Sampling and provenance

Each simulation contributes the first trajectory and its middle stored time step. The downloader pins the current upstream repository commit before resolving a file. Within the first test-file listing it prefers the documented Gray–Scott parameter pair or an EIT flow filename when present, then smaller containers. It reads an explicitly named scalar array using [The Well's dimension metadata](https://polymathic-ai.org/the_well/data_format/); it does not guess vector components or extract 2D slices from 3D fields. Missing scalars, unsupported coordinates, nonfinite samples, and effectively constant prepared snapshots are rejected.

Preparation averages scalar values over overlapping equal-area cells to reach at most **100 × 100**. Fractional overlaps handle noninteger reduction factors. It does not crop, upsample, or normalize scalar values. Averaging smooths fine features and can change extrema, saddles, and answers: the prepared array is the authoritative task field. Display interpolation happens afterward. The existing 101 × 101 terrain is likewise averaged to 100 × 100 at app startup; its source file remains intact.

New datasets use **normalized raster axes [0, 1] × [0, 1]**, matching the app's square-grid contract. Physical coordinates are retained in provenance but are not used as plotted axis units. In particular, a physically rectangular simulation domain is shown in normalized raster coordinates; gradients and areas are calculated in these normalized units. Arrays are oriented with row zero at lowest y.

Each prepared field has its source URL and citation, license, selected scalar, trajectory/time index, source and prepared dimensions, preparation method, and checksums recorded. Simulation records also contain the immutable upstream commit and source coordinates. The app validates prepared-file checksums and carries this provenance into generated task exports, including `everything.zip`.

Multiple scalar arrays per dataset remain future work. These examples deliberately select one documented scalar per source; users cannot yet choose among arrays within an uploaded container.

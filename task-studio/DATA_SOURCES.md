# Dataset sources and attribution

The standalone application bundles the same 12 prepared inputs as the research app. Arrays and their sidecar JSON files are included; large upstream originals are not needed. The sidecars preserve source URLs, citations, checksums, selected variables, dimensions, and preparation details. Historical research paths inside provenance identify acquisition history and are not runtime dependencies.

| Input | Source and citation | Recorded reuse terms |
| --- | --- | --- |
| Terrain | Existing 101 × 101 DEM extract, averaged to 100 × 100 at startup; original coordinates bundled in `task_app/builtin/`. | Upstream DEM attribution/license not recorded in this extract. |
| Monotone and plateau controls | Project-generated synthetic controls (`s01.npy`, `s02.npy`). | Project authors. |
| Acoustic scattering · pressure | [The Well; acoustic scattering, Michael McCabe; Clawpack, Mandli et al. (2016).](https://polymathic-ai.org/the_well/datasets/acoustic_scattering_discontinuous/) | CC-BY-4.0 |
| Active matter · concentration | [Maddu, Weady & Shelley (2024), Learning fast, accurate, and stable closures of a kinetic theory of an active fluid. The Well.](https://polymathic-ai.org/the_well/datasets/active_matter/) | CC-BY-4.0 |
| CCC globe (Africa) · screenshot intensity | [Nardini et al., A Testing Environment for Continuous Colormaps, IEEE TVCG 27(2), 1043–1053 (2021; online 2020), doi:10.1109/TVCG.2020.3028955.](https://ccctool.com/) | Original image/data reuse terms not established from the supplied screenshot. |
| CCC globe (South Pole) · screenshot intensity | [Nardini et al., A Testing Environment for Continuous Colormaps, IEEE TVCG 27(2), 1043–1053 (2021; online 2020), doi:10.1109/TVCG.2020.3028955.](https://ccctool.com/) | Original image/data reuse terms not established from the supplied screenshot. |
| CCC mammogram · screenshot intensity | [Nardini et al., A Testing Environment for Continuous Colormaps, IEEE TVCG 27(2), 1043–1053 (2021; online 2020), doi:10.1109/TVCG.2020.3028955.](https://ccctool.com/) | Original image/data reuse terms not established from the supplied screenshot. |
| Euler gas dynamics · density | [The Well; Euler multi-quadrant Riemann problems, Marsha Berger, Ruben Ohana & Michael McCabe; Clawpack.](https://polymathic-ai.org/the_well/datasets/euler_multi_quadrants_openBC/) | CC-BY-4.0 |
| Gray–Scott · species B concentration | [The Well (Ohana et al., 2024); Gray–Scott dataset, Daniel Fortunato.](https://polymathic-ai.org/the_well/datasets/gray_scott_reaction_diffusion/) | CC-BY-4.0 |
| Human cell nuclei · fluorescence intensity | [Moffat et al. (2006), doi:10.1016/j.cell.2006.01.040; image copyright David Root, CC0.](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.human_mitosis) | CC0 |
| Retinal lesions · green-channel intensity | [Budai et al. (2013), doi:10.1155/2013/154860. Published HRF image 07_dr.JPG crop, green channel; CC0 from Andreas Maier.](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.microaneurysms) | CC0 |

The Well sources are attributed in each sidecar; the arrays were reduced by area-weighted averaging to at most 100 × 100 and may differ in extrema/topology from their originals. The biomedical sources are stored image intensities, not calibrated clinical measurements.

The three CCC inputs are derived from user-supplied screenshots of CCC-Tool. They retain screen intensities, square padding, and documented background repair; they are not reconstructions of original scientific values. The original dataset identity and image-reuse terms are not established by those screenshots. The two globe views may show the same simulation. These attribution gaps are preserved explicitly; this bundle does not claim that all datasets share an open license.

CCC reference: Nardini et al., *A Testing Environment for Continuous Colormaps*, IEEE TVCG 27(2), 1043–1053 (2021; online 2020), [doi:10.1109/TVCG.2020.3028955](https://doi.org/10.1109/TVCG.2020.3028955). The [SciVis catalog](https://github.com/huyen-nguyen/scivis-catalog) guided source discovery; its future paper citation can be added here when available.

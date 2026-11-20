# Scalar Task Studio

Select pools of datasets, colormaps, operations, selections, aggregations, and actions; generate tasks and preview their illustrated table immediately. See [README.md](README.md) for local startup, GitHub packaging, deployment, sessions, and public-demo limits.

## Add scientific examples

Three supplied CCC screenshots are also available as 100 × 100 image-intensity fields: two globe views and a mammogram. Restart the app to load them. These bring the current picker to twelve inputs. They contain display intensities rather than recovered scientific measurements. See [dataset provenance](DATA_SOURCES.md).

Run `uv run --locked --extra datasets python -m task_app.dataset_download` from the repository root, then restart the app. This prepares seven independent sources at at most 100 × 100, with a 64 MiB total data-transfer cap and a preview sheet. See [DATASETS.md](DATASETS.md) for the selection, provenance, download limits, and validation status.

## Downloads

- **tasks.csv**: one task per row, including the sampled stages, concrete parameters, image reference, exact prompt, answer specification, complete ground truth, context, and provenance. Structured cells are JSON. Uploaded names starting with spreadsheet formula characters receive a protective apostrophe in CSV only.
- **images.zip**: PNG files under `images/`. Extract beside `tasks.md`.
- **tasks.md**: illustrated table modeled after the project's `task_example.md`, including prompts and answers.
- **everything.zip**: CSV, Markdown, PNGs, tasks.json, manifest.json, and source arrays/colormaps in `inputs/`. Extract together to preserve relative image links. This is an authoring/review export and includes correct answers.

The **Sampled parameters** column in the preview and Markdown shows only independently sampled choices: minimum/maximum, extrema scope, thresholds, point locations, box bounds, window width and size cue, interval measure, or correlation direction. Box sizes are omitted because they follow from the bounds. Square windows show only the sampled width in cells; their height and physical side length are derived. A task without extra parameter draws says “None beyond the selected stages.” Coordinates are expressed in the plotted axis units. Display numbers use 10 significant digits; CSV/JSON preserve the full numeric values.

In CSV and JSON, `parameters` contains only those sampled choices. Fixed rendering/calculation conventions and derived bookkeeping are stored separately in `context` (grid geometry, interpolation, connectivity, equal object weighting, selected object counts, comparison equality epsilon, physical window side length, and fixed reference-square placement). JSON also includes `parameter_summary`, the readable lines used in the preview and Markdown. Previously generated batches retain their original schema; restart and generate a new batch for this layout.

## Upload formats

Datasets: CSV numeric matrix without header, or NPY numeric array without pickle. Finite square grids from 3×3 to 128×128, absolute values ≤1e100. Specify the shared x/y axis bounds. Cells are equal squares; row zero is lowest y. No geographic or missing-data interpretation is inferred.

Colormaps: JSON list of colors (`["#172554", "#38bdf8", "#fef08a"]`), JSON object with a `colors` list, or RGB/RGBA CSV without header. Numeric colors use 0–1 or 0–255, with a common scale for all channels. Between 2 and 4096 opaque colors. Interpolated in the supplied order. Upload limit 2.5 MB in the UI.

Built-ins: the saved terrain in `task_app/builtin/` (copied from the research terrain), now averaged from 101×101 to 100×100, the monotone and plateau controls in the same folder, and the colormaps listed below. The averaged 20×20 terrain is no longer offered. The saved terrain is itself an earlier crop/subsample of the original DEM, not the complete original DEM. Its displayed bounds remain −0.015 to 3.015; reduction averages overlapping cells over that same domain. Original files remain unchanged. Downloaded examples are also at most 100×100 and use normalized raster axes [0,1]. Prepared arrays, rather than the higher-resolution upstream originals, define task answers. Uploaded arrays retain their supplied resolution. Corrupt prepared files are skipped with a warning in the picker. Each task uses linear color normalization over its original displayed-field range. Gradient tasks also display the original scalar field and its original color scale. Selections, statistics, and answers still use gradient magnitude; the prompt states this explicitly. Task context distinguishes the displayed field and color range from the analysis field and its range, and records the existing incident-edge gradient convention (computed before selection). Prompts use direct point/region wording; sampling conventions remain in metadata.

### Built-in colormaps

The picker includes gray, viridis, cividis, coolwarm, RdBu, turbo, jet, twilight, prism, gist_ncar, plasma, inferno, and magma, in that order. Gray is the only default selection. Each map uses 256 sampled colors and is available for task generation, the parameter-space counts, and exported inputs. These use the existing Matplotlib dependency; see its [colormap reference](https://matplotlib.org/stable/users/explain/colors/colormaps.html).

### Interpolation

The app uses **bilinear scalar interpolation before color mapping**, following the ordering of [PyVista's `interpolate_before_map=True`](https://docs.pyvista.org/examples/02-plot/interpolate_before_map.html). `interpolation.py` supplies one evaluation routine for point queries and for the display-only 768×768 scalar raster. Colormapping happens after that raster is evaluated; there is no interpolation of mapped RGB colors. This is the regular-grid bilinear choice, not a claim of pixel-identical VTK surface rendering, whose interpolation depends on mesh topology.

The selected task array uses sample-centered positions. Prepared examples have already been reduced by the downloader (or, for terrain, at app startup). The outer half-cell uses the nearest boundary sample. Interpolation does not upsample or overwrite the selected task array. Region statistics, gradients, topology, and sampled click candidates still use task-grid samples, rather than the display raster. Point questions currently place their markers at task sample positions; their answers use the same interpolation method, evaluated on the analysis field (the original scalar field or its gradient magnitude). The helper also supports arbitrary in-domain positions. Existing click acceptance remains based on task grid cells. Interpolation settings and the statistics support are recorded in each task; source arrays and their geometry are included in the complete ZIP. Previously generated batches are preserved; restart the app and generate a new batch for interpolated images.

## Composition and sampling

`contracts.py` propagates kind, group cardinality, singleton status, retained locations, and whether objects carry scalar values. It does not enumerate allowed recipes. The server is authoritative; unknown options and incompatible chains are rejected. Options remain toggleable so users can build combinations in any order. A checked option with no current compatible path has an explanatory tooltip.

The **Compatible stage combinations** panel defaults to **Visualization**, with one colored spline through Operation → Selection → Aggregation → Action for each included tuple. Excluded curves are faint and dashed. Hover or focus a curve to trace its stages; click it, or press Enter/Space, to toggle inclusion. The inspector selector lets you highlight and toggle combinations when curves overlap. Switch to **List** for the checkbox table. Both views share the same exclusions, plus Include all listed / Exclude all listed controls. Each exclusion applies across the selected datasets and colormaps. Exclusions survive column changes within the current page, including temporarily hiding a combination; refreshing the page resets them. Included counts and parameter-space totals update immediately. If every compatible row is excluded, generation is disabled. The backend also enforces exclusions, and `manifest.json` records their stable IDs in `configuration.excluded_combinations`, plus the compatible and included counts.

Single-number inputs only allow **No reduction**. Two marked points remain two singleton groups: they can be compared directly but cannot each be aggregated. If a sampled rectangle or filter produces exactly one sample, aggregation rejects that parameter draw and resamples it. Empty-set behavior remains defined separately for area, components, count, and existence.

The compatibility display also reports nominal continuous parameter dimensions and discrete setting counts, with a per-dataset/stage breakdown. These are upper bounds before validity filtering. Grid positions and rectangles are discrete in the current generator; thresholds are continuous (one dimension for above/below, two for between thresholds). Counts include the selected datasets and colormaps.

Each requested slot samples uniformly from included compatible stage tuples, then independently samples a dataset and colormap. All discrete parameters are uniform. Rectangles sample independent pairs of distinct grid boundaries; points sample distinct grid cells; windows sample odd widths from 3 through grid width; above/below thresholds sample continuously between analysis-field minimum and maximum; between-threshold selections draw two independent uniform values in that range and sort them as lower and upper bounds. Parameter draws are conditioned on data-dependent validity. Equal per-option marginals are not guaranteed because options participate in different numbers of compatible tuples. Duplicate tasks are excluded within each batch. A content fingerprint includes source-data and colormap hashes, stages, concrete parameters, context, prompt and answer; seeds, generated IDs, and asset names do not make a task unique. Different colormaps count as distinct tasks. Duplicate candidates trigger a new uniform candidate draw, with a limit of 100 candidate draws per requested slot. A shortfall after that limit is reported, not proof that the entire space is exhausted.

Up to 50 parameter draws are attempted within each selected tuple and dataset. Invalid slots are reported with their reasons and are not silently replaced. Thus a batch may contain fewer tasks than requested. Constant channels make correlation undefined; ratios between two groups require positive operands; click tasks need accepted locations; scalar summaries need nonempty groups (area may be zero). Counts/existence allow empty groups. Tied answers are retained.

The manifest records the configuration, parameter domains, failures, seed, version, actual count, and number of rejected duplicates. Each task records its own seed, attempt number, dataset/colormap hashes, full parameters, context, and image hash. A task can be reproduced with the exported NPY/colormap and the recorded code version and seeds. PNG byte identity can depend on Matplotlib/font versions.

Tasks provide reference answers and their data types, without a scoring policy, response-error tolerances, or a multiple-choice setup. Users can choose multiple choice, exact matching, or their own error metric later. Ground-truth calculation still records its numerical equality epsilon for P/Q/Equal comparisons; this handles floating-point ties and does not grade user responses. Pearson classification uses ±0.1, population SD uses ddof=0, threshold components use 4-connectivity, and extrema plateaus use 8-connectivity. These conventions define the task and its answer. Previously generated batches retain their original fields; generate a new batch for exports without scoring.

This version exposes individual aggregation stages and window-statistic extrema through Find. It does not yet expose arbitrary aggregation chains, paths between endpoints, custom units, masks, rectangular grids, or multi-channel uploads. These need additional contracts and rendering conventions.

## Checks

```sh
uv run --locked --extra datasets python -m unittest discover -s tests -v
```

The dependency-free SVG component also has geometry and interaction checks: `node tests/test_visualization.mjs` (when Node is installed).

Threshold selections use strict `value > threshold` / `value < threshold` for above/below. Between thresholds uses `lower ≤ value ≤ upper`, including both endpoints. Its interval width varies: there is no fixed target tolerance. The former Cells near target value option is replaced by Cells between thresholds; previous exported batches are preserved.

### Means of extrema and saddle points

Extrema and saddle selections carry scalar values, so **Mean → Retrieve** is compatible. Extrema means give one equal-weight contribution per connected plateau, regardless of plateau area. Saddle means give one contribution per saddle. Both require at least two selected objects; smaller sets trigger parameter retries and, if no valid task is found, a reported shortfall. No reduction still supports count, existence, and find. Saddle points also support **No reduction → Retrieve coordinates**: the response is one `[x, y]` pair, and any detected saddle is accepted. At least one saddle must exist. The export retains every valid coordinate pair as ground truth, without prescribing how a user response should be scored.

Extrema have the same allowed aggregation/action combinations as saddles: **No reduction → Retrieve coordinates, Find + click, Count, Test existence, Absolute difference, or Ratio**, and **Mean → Retrieve values**. Both also support Range, Extent X, and Extent Y followed by Retrieve values, Absolute difference, or Ratio. Compare order remains unavailable for these unpaired sets. Extrema coordinate tasks accept the center of any cell in any qualifying plateau, using the sampled minimum/maximum and all/non-global/global scope. At least one qualifying extremum must exist; all qualifying coordinates are exported as ground truth. Plateau cells remain one object for counting, means, differences, and ratios.

Saddles are the nondegenerate stationary points **strictly inside bilinear patches**, using the same sample-centered geometry as scalar interpolation. For a patch `f(u,v) = a + b*u + c*v + d*u*v`, the detector solves `u = -c/d`, `v = -b/d` and keeps solutions strictly inside `(0,1)²` with `d != 0`; the Hessian determinant is negative. Grid-edge/vertex saddle cases and degenerate stationary sets are excluded, as stated in prompts and task metadata. This is an explicit subset of critical points, not a complete classification of every grid vertex. Selected task-array values are never rounded or resampled during detection; preparation may already have reduced a downloaded source. Saddle click answers use coordinate distance (one tenth of sample spacing), because their positions generally lie between samples. No new random parameters are needed for saddle selection.


### Window-size cues

Every window task uniformly samples a discrete two-choice parameter, `window_size_cue`: **axis_scale** or **reference_square**. The first states the side length in plotted axis units and draws a matching ruler below the field. The second draws a transparent dashed square R at the required size and refers to it in the prompt. Both cues use `window_side_length = window_cells × cell_spacing`; the statistic and accepted window centers are identical for a fixed field, width, and extremum. Reference-square placement is fixed and independent of the answer. The parameter-space count includes the factor of two, and all sampled cue settings and reference bounds are exported. Prompts no longer require the viewer to infer a window size from invisible grid cells.

Saddle selection returns the complete detected set, which can contain zero, one, or many points. There is no special two-saddle selection. Two-way Compare order is incompatible with this unpaired set.

**Saddle points → No reduction → Absolute difference / Ratio** acts on the complete detected set. Difference is highest minus lowest scalar value; ratio is highest divided by lowest. Both return the literal `undefined` when fewer than two saddles exist, and ratio also returns `undefined` if the lowest value is zero. Negative nonzero denominators retain their signed mathematical ratio. Two or more equal-valued saddles have difference zero (and ratio one if the shared value is nonzero). These are valid tasks, not failed samples. The same scalar-valued-object contract supports extrema sets. Ground truth records a number or the string `undefined`; no grading rule is attached. JSON, CSV, and Markdown preserve it as the string `undefined`. The selection label is Extrema.


### Range and coordinate extents

**Range** returns `[minimum value, maximum value]`. **Extent X** and **Extent Y** return `[minimum coordinate, maximum coordinate]` on their respective axes. Extents use selected cell centers, all member-cell centers for selected extrema plateaus, and interpolated saddle positions. They do not use the outer edges of cells. The operation determines the analysis field and any threshold or critical-point selection; the image always displays the original scalar field.

For one selected group, **Retrieve values** returns the ordered pair; **Absolute difference** returns maximum minus minimum; **Ratio** returns maximum divided by minimum. An endpoint ratio is `undefined` when the minimum is zero and otherwise retains its mathematical sign. These interval aggregations require at least two selected cells or objects. Existing No reduction actions on extrema/saddles retain their earlier empty/singleton semantics.

For two groups, the action uniformly samples `interval_measure` from **minimum / maximum / width** and compares that same measure between groups. Width is maximum minus minimum. For windows, the same three-choice parameter selects the measure and the existing binary `find_extremum` selects its minimum or maximum. Both independently sampled choices appear in Parameters and the parameter-space counts. Window width, cue, and placement constraints still apply. Spatial widths are computed in grid-spacing units for grid-based selections so equal-width windows remain tied despite floating-point endpoint rounding.

Pair answers use `answer_spec.type = number_pair`, with explicit length and minimum/maximum order. Pair answers are distinct from `[x, y]` coordinate responses. CSV, JSON, Markdown, and the preview preserve both endpoints. Version 2 changes Range from its former scalar width to an endpoint pair; earlier batches remain unchanged.

### Area and the whole-field reference

The separate **Area fraction** aggregation has been removed. **Area → Retrieve values** returns selected area in squared coordinate units. When there is one selected group, **Area → Ratio** returns selected area divided by the whole field area, and **Area → Absolute difference** returns the absolute area difference from the whole field. For example, an area of 6 in a field of area 25 produces ratio 0.24 and difference 19. Empty selections produce ratio 0 and difference equal to the field area; the whole-field selection produces ratio 1 and difference 0.

The reference is fixed, so it is recorded as `context.comparison_reference = whole_field_area` and `context.reference_area`, not as a sampled parameter. Two rectangles and inside/outside selections still compare their two selected areas directly. Extent X/Y retain the endpoint-pair rules above; they receive no implicit whole-domain comparison. The Ratio action label is now simply Ratio, with the exact numerator and denominator stated in each prompt.

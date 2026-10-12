# Public demo configuration

## Scientific separation

Every curriculum record, cohort, profile, score, admission route, reference series, and demo-specific coefficient in this package is fictitious. The configuration exists to exercise the generic generator and should not be interpreted as a calibrated representation of any organization.

## Parameter classes

The input workbooks classify every exposed parameter in a `clasificacion` column.

### `GENERIC_ENGINE_PARAMETER`

These values define stable software mechanics or explicit data rules: grade scale 1–7, passing threshold 4.0, one-decimal half-up rounding, rescaled-Beta grade family, course-attempt rules, no replacement of approval by an independent probability, load-selection rule, terminal-event switches, distribution family labels, and the neutral-admission-signal convention.

### `DEMO_PARAMETER`

These values are intentionally fictitious: course-level means and dispersions, temporal persistence and variation, concentration factor, cohort sizes, simulation horizon, course loads, heterogeneity scales, admission effect, prior-performance effects, repeat-attempt effects, non-continuity hazard, school-background probabilities, development seed, and demonstration-seed range.

### `INSTITUTION_CALIBRATED_PARAMETER`

No parameter with this classification is distributed in the public package. Any value that would require observed institutional evidence has been replaced by a demo value and marked `DEMO_PARAMETER`.

## Demo design

- Curriculum: 50 courses named `Course 001` through `Course 050`.
- Graph: ten levels, six roots, a new prerequisite topology, and one terminal course.
- Reference periods: 2031-01 through 2044-02.
- Simulation periods: 2040-01 through 2044-02.
- Active cohorts: 2040 through 2044.
- Admission routes: `REGULAR`, `ADMISSION_A`, `ADMISSION_B`, and `ADMISSION_C`.
- Admission profiles: 300 records with IDs `PROFILE_0001` onward.
- Development seed: 42.

These choices are demonstration controls, not estimates.

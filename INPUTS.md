# Authorised analysis inputs

Inputs are obtained through the study's data-access arrangements and remain outside the software repository. Source registries, patient-specific cleaning decisions and identifier mappings are managed by the data custodian. The analysis entry point accepts prepared extracts and does not perform source-record adjudication.

## Required files

| File | Purpose |
|---|---|
| `cohort.csv` | Development and temporal cohorts |
| `cohort_multiple_episode_sensitivity.csv` | Episode-retention sensitivity using the same field schema |
| `cohort_icu_sensitivity.csv` | ICU-availability sensitivity using the same field schema |
| `legacy_history.csv` | Historical feature inventory, linked by `study_id` |
| `clinical_spec.json` | Privately supplied screening metadata |
| `input_manifest.json` | SHA-256 identity of every supplied input file |

`multicentre_collection.json` is optional. When supplied, its `external_hospitals` and `registry_records` fields populate the collection-status panel. Without it, the panel shows the planned evaluation workflow without collection totals. Each preparation verifies the current inputs, then removes any staged copy of an optional file omitted from the current bundle, including when reusing an output directory.

## Cohort fields

The three cohort files require a unique pseudonymous `study_id`, a `cohort` value of `development` or `temporal`, nominal `mrs6`, `mrs12` and `mrs24` scores, and the clinical fields used by the analyses:

`age`, `female`, `nmdar`, `lgi1`, `worst_mrs`, `icu_admission`, `respiratory_failure`, `impaired_consciousness`, `movement_abnormality`, `log1p_treatment_delay`, `preceding_icu`, `treatment_delay_days`, `neoplasm`, `mog`, `antibody_codes`, `legacy_multiple`, `tx_steroid`, `tx_ivig`, `tx_plasma_exchange` and `tx_immunosuppressant`.

mRS values range from zero to six; unavailable assessments are empty. Known death remains absorbing in the longitudinal flow. Binary fields use zero or one with empty values for missing observations. `antibody_codes` contains pipe-separated integer codes defined in `configs/antibody_codes.json`.

The historical extract uses `study_id`, `recovery_order`, the fields listed in `configs/legacy_spec.json`, and their source components. The source components and transformations are defined in that specification and `code/legacy_omissions.py`. `configs/feature_dictionary.json` contains variable definitions, not clinical observations.

## Screening metadata

`clinical_spec.json` contains a `screening` object with `internal_screened`, `no_immunotherapy`, `infectious_encephalitis`, `below_age_14`, `inconsistent_internal_timeline` and `missing_qualifying_treatment_date`. These counts are supplied by the data custodian and checked against the runtime cohort totals. No study counts are embedded in the distributed configuration.

An `antibody_codes` mapping may also be supplied. Otherwise the software uses the distributed code definitions.

## Input identity

`input_manifest.json` contains a `files` list. Each entry has a relative `file` name and its `sha256` digest. The entry point verifies the required CSVs, the screening metadata and any supplied collection metadata against that manifest before analysis.

The authorised input bundle, its manifest, patient linkage keys, numerical outputs and fitted objects are kept outside the repository. The software licence does not grant access to those materials.

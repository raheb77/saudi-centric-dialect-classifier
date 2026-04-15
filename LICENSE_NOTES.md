# License Notes

## Scope
This file records the licensing and usage constraints visible in the local source packages under `data/raw/`. It is operational guidance for this repository, not legal advice.

## Observed Local License Files
The inspected local NADI license files all state `CC BY-NC-ND 4.0`:

- `data/raw/nadi2020/NADI_release/NADI-Twitter-Corpus-License.txt`
- `data/raw/nadi2021/NADI2021_DEV.1.0/NADI2021-Twitter-Corpus-License.txt`
- `data/raw/nadi2023/NADI2023_Release_Train/NADI2023-Twitter-Corpus-License.txt`
- `data/raw/nadi2023/test/NADI2023_Release_Test/NADI2023-Twitter-Corpus-License.txt`

## Source-by-Source Notes

### NADI 2020
- Local license file declares `CC BY-NC-ND 4.0`.
- The local release README also states that shared-task participants should not share the distributed tweets outside their labs or publish them publicly.
- Operational takeaway for this repo: keep the raw package local, do not republish raw tweet text, and keep any future public outputs at the metadata, documentation, or aggregated-results level unless a separate review says otherwise.

### NADI 2021
- Local license file declares `CC BY-NC-ND 4.0`.
- The local release README repeats the shared-task restrictions on additional tweet use, outside information, and public redistribution of the distributed tweets.
- Operational takeaway: treat the local NADI 2021 DA files as research inputs inside this repo, not as text to be repackaged and redistributed.

### NADI 2023
- Local license file declares `CC BY-NC-ND 4.0`.
- The local release README positions Subtask 1 as a closed track and restricts use to the provided datasets for the shared task itself.
- Operational takeaway for this repo: use the local files for internal benchmarking and documentation, but do not interpret the bundled data as permission to redistribute derived tweet-text corpora publicly.

### MADAR-2018
- The local repo contains `data/raw/nadi2023/NADI2023_Release_Train/Subtask1/MADAR-2018.tsv` as part of the NADI 2023 package.
- This repository does not currently assert a separate standalone MADAR license beyond what is visible in the local NADI 2023 package.
- Operational takeaway: keep MADAR documented as a bundled reference / future OOD source and avoid making separate redistribution claims until its standalone licensing is reviewed directly.

## Repository Rules Derived from the Local Licenses
- Do not modify raw files in place under `data/raw/`.
- Do not automatically download or hydrate new data from external services.
- Do not publish raw tweet text copied from the local NADI packages.
- Keep documentation, manifests, and future experiment outputs reproducible and source-attributed.
- Assume that any future release of derived text datasets requires an explicit license review because `ND` can limit redistribution of adapted text.

## Citation Expectations
If the local NADI sources are used in experiments or reports, cite the relevant shared task release:

- NADI 2020 shared task
- NADI 2021 shared task
- NADI 2023 shared task

The local license files already include citation guidance. Repository-level documentation should reference the releases by name and year and should not invent unsupported bibliographic details.

## Practical Implication for This Repository
This repo can safely contain:

- raw-source inventory metadata
- documentation about scope and annotation rules
- future model cards and aggregate metrics
- code that reads local data without modifying raw files

This repo should avoid publishing:

- copied raw tweet dumps
- hydrated tweet text obtained from unlabeled ID files
- redistributed transformed text corpora without a separate review

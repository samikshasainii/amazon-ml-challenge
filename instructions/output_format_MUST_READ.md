inal Submission Package:
In addition to your live leaderboard uploads, every team submits a single zip archive with your code and outputs. We use it to reproduce your results, audit your blocking, and check the fair-play and model-license rules — the top teams' packages are reviewed in detail before the final rankings are confirmed.

Structure:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (same file you upload to the leaderboard)
│   └── candidate_pairs.tsv         # your blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all your source code
│       ├── README.md               # how to reproduce end-to-end (data → blocking → matching → output)
│       └── requirements.txt        # pinned dependencies / environment
└── Documentation_template.md       # your methodology write-up (this filled-in template)
output/ — the two TSV files described above: matching_results.tsv and candidate_pairs.tsv.
code/business_entity_resolution/ — a self-contained, runnable copy of your pipeline. Put all source under src/, and include a README.md with exact run instructions plus a requirements.txt (or equivalent environment file) pinning versions. Anyone should be able to regenerate both output files from the training/test data using only what is in this folder.
Methodology document — fill in the provided Documentation_template.md and drop it straight into the zip (the filled-in .md is fine; a .pdf export works too). No need to rename it.


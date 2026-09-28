# Input data

The input data can be downloaded: 

```text
FeatureDataWoOut.pkl
```

## Expected organization

- Rows: microbiome samples
- Feature columns: CLR-normalized ASV or taxon abundance features
- Final six columns: sample metadata
- Required target columns: `Country`, `Continent`, `Grape_variety`, `Rootstock`, and `comb`

The scripts follow the original analysis assumption that the final six columns are metadata and all preceding columns are numeric microbial features. If your deposited file uses a different layout, update `N_METADATA_COLUMNS` in `scripts/analysis_config.py`.

The input dataset is intentionally not synthesized or fabricated in this package. Add the actual manuscript analysis file, subject to the applicable data-sharing permissions, before making the repository public. If the file is too large for normal Git hosting, use Git LFS or provide a permanent repository link and checksum here.

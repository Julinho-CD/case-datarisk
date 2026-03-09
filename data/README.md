# Data Policy

This public repository does not redistribute the original Datarisk challenge data.

## Official Source

- Official repository: <https://github.com/datarisk-io/datarisk-case-ds-junior>
- Runtime download source: official raw CSV files from that repository

## Local Files Expected

If you want fully local execution without downloading at runtime, place these files under `data/raw/`:

- `base_cadastral.csv`
- `base_info.csv`
- `base_pagamentos_desenvolvimento.csv`
- `base_pagamentos_teste.csv`

## Runtime Behavior

- Local mode: use the CSVs already present in `data/raw/`.
- Public demo mode: download the official CSVs and cache them under `.cache/`.
- Processed outputs may be created locally for convenience, but they are not part of the public repository.

## Local Structure

```text
data/
+-- raw/
|   +-- base_cadastral.csv
|   +-- base_info.csv
|   +-- base_pagamentos_desenvolvimento.csv
|   +-- base_pagamentos_teste.csv
+-- processed/
```

## Local Command

```bash
python -m src.make_dataset
```

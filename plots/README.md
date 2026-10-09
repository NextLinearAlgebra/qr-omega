# Figures and numbers of the paper

`reproduce.py` regenerates the figures of the paper and the numbers it quotes from the recorded
measurements; it needs no GPU and runs in seconds:

```sh
python3 -m venv .venv && .venv/bin/pip install -r plots/requirements.txt
.venv/bin/python plots/reproduce.py
```

With the versions of `requirements.txt` the figures are bit for bit those of the paper
(`tests/test_figures.py` checks this).

| File | Content |
| --- | --- |
| `../reproducers/presets.json` | QR-Ω's measured cells |
| `data/references.csv` | The measurements of cuSOLVER (cuSOLVERMp on several GPUs), MAGMA, SLATE and GEMM, one row per cell: the fastest tuned configuration and its numerical checks |
| `data/accuracy8-provenance.json` | For the eight-GPU reference cells: the validated run of the timed configuration that supplies their errors, with the hash of its record |
| `data/paper.csv` | The table of all measurements that the figures are drawn from, assembled by `reproduce.py` |
| `figures/fig*.{pdf,png}` | The figures, named by their number in the paper |
| `figures/paper-numbers.csv` | Every number the paper quotes next to the same quantity computed from the data: `exact` when both print alike, `rounding` when they differ by at most one unit of the last printed digit, `differs` otherwise |
| `figures/manifest.json` | Hashes of the inputs and the list of figures |

`measurements.py` assembles and checks the table, `figures.py` draws the figures and
`paper_numbers.py` recomputes the quoted numbers.

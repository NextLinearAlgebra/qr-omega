"""CSV schema shared by the publication plot renderers."""
import csv

FIELDS = ["id", "kind", "library", "precision", "gpus", "m", "n", "time_s", "tflops", "residual", "orthogonality", "validation", "residual_metric", "orthogonality_metric", "checked_columns", "statistic_block_columns", "repetitions", "timing_statistic", "boundary", "host", "gpu_uuid", "configuration", "binary_sha256", "runtime_audit", "strict_integrity", "strict_production", "source", "source_sha256", "receipt", "notes"]

def dump_csv(path, rows, fields=FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

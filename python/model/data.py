import csv
import math
from pathlib import Path

import numpy as np

from .errors import PPIxGPNError

TARGETS = ("Abeta", "GFAP", "NfL", "pTau")
FIXED_COLUMNS = ("record", "id", "split") + tuple(f"Y_{target}" for target in TARGETS)
SPLITS = ("train", "valid", "test")
DEFAULT_FILE = Path(__file__).resolve().parents[2] / "dataset" / "sample.csv"


def load_dataset(file=None):
    path = DEFAULT_FILE if file is None else Path(file).expanduser()
    if not path.is_file():
        raise PPIxGPNError("PPIxGPN:MissingFile", f"Dataset file does not exist: {path}")
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = [row for row in csv.reader(stream) if any(cell.strip() for cell in row) or len(row) > 1]
    except (UnicodeDecodeError, csv.Error) as error:
        raise PPIxGPNError("PPIxGPN:InvalidDataset", f"Cannot read {path}: {error}") from error
    if not rows:
        raise PPIxGPNError("PPIxGPN:InvalidDataset", f"Dataset file is empty: {path}")

    header = [cell.strip() for cell in rows[0]]
    num_fixed = len(FIXED_COLUMNS)
    if len(header) <= num_fixed or tuple(header[:num_fixed]) != FIXED_COLUMNS:
        raise PPIxGPNError("PPIxGPN:InvalidDataset", f"The header must start with {','.join(FIXED_COLUMNS)} "
                           "and continue with protein columns.")
    protein = header[num_fixed:]
    if len(set(protein)) != len(protein) or "" in protein:
        raise PPIxGPNError("PPIxGPN:InvalidDataset", "Protein column names must be unique and nonempty.")

    width = len(header)
    text = []
    value = np.full((len(rows) - 1, width - 3), np.nan)
    for position, row in enumerate(rows[1:]):
        if len(row) > width:
            raise PPIxGPNError("PPIxGPN:InvalidDataset",
                               f"Data row {position + 1} has {len(row)} fields, but the header defines {width}.")
        cells = [cell.strip() for cell in row] + [""] * (width - len(row))
        text.append(cells[:3])
        value[position] = [_number(cell) for cell in cells[3:]]
    text = np.array(text, dtype=str).reshape(-1, 3)

    is_participant = text[:, 0] == "participant"
    is_ppi = text[:, 0] == "ppi"
    if not (is_participant | is_ppi).all():
        raise PPIxGPNError("PPIxGPN:InvalidDataset", "The record column may contain only participant and ppi.")
    if not is_participant.any():
        raise PPIxGPNError("PPIxGPN:InvalidDataset", "The dataset contains no participant rows.")

    participant = text[is_participant, 1]
    if (participant == "").any() or np.unique(participant).size != participant.size:
        raise PPIxGPNError("PPIxGPN:InvalidDataset", "Participant identifiers must be unique and nonempty.")
    partition = np.char.lower(text[is_participant, 2])
    if not np.isin(partition, SPLITS).all():
        raise PPIxGPNError("PPIxGPN:InvalidDataset", "Each participant split must be train, valid, or test.")
    Ydata = value[is_participant, :len(TARGETS)]
    if not ((Ydata == 0) | (Ydata == 1)).all():
        raise PPIxGPNError("PPIxGPN:InvalidDataset", "Diagnosis labels must be 0 or 1 for every participant.")
    Xdata = np.ascontiguousarray(value[is_participant, len(TARGETS):].T)
    if not np.isfinite(Xdata).all():
        raise PPIxGPNError("PPIxGPN:InvalidDataset", "Protein expression must be finite for every participant.")

    network_id = text[is_ppi, 1].tolist()
    location = {name: position for position, name in enumerate(network_id)}
    if len(network_id) != len(protein) or len(location) != len(network_id) or not set(protein) <= set(location):
        raise PPIxGPNError("PPIxGPN:InvalidDataset", "Provide exactly one ppi row for each protein column.")
    ppi_data = value[is_ppi, len(TARGETS):][[location[name] for name in protein]]
    if not np.isfinite(ppi_data).all():
        raise PPIxGPNError("PPIxGPN:InvalidDataset", "PPI scores must be finite.")

    return {
        "Xdata": Xdata,
        "Ydata": Ydata,
        "ppi_data": ppi_data,
        "protein": np.array(protein, dtype=str),
        "participant": participant,
        "split": partition,
        "target": np.array(TARGETS, dtype=str),
        "idx_train": np.flatnonzero(partition == "train"),
        "idx_valid": np.flatnonzero(partition == "valid"),
        "idx_test": np.flatnonzero(partition == "test"),
        "source": str(path),
    }


def _number(text):
    try:
        return float(text) if text else math.nan
    except ValueError:
        return math.nan

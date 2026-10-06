import csv
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .errors import PPIxGPNError


class Table(Mapping):
    def __init__(self, columns):
        converted = {}
        for name, value in dict(columns).items():
            array = np.asarray(value)
            if array.ndim != 1:
                raise PPIxGPNError("PPIxGPN:InvalidTable", f"Column {name} must be one-dimensional.")
            converted[str(name)] = array
        if len({array.size for array in converted.values()}) > 1:
            raise PPIxGPNError("PPIxGPN:InvalidTable", "All table columns must have the same length.")
        self._columns = converted

    def __getitem__(self, name):
        return self._columns[name]

    def __iter__(self):
        return iter(self._columns)

    def __len__(self):
        return len(self._columns)

    def __eq__(self, other):
        if not isinstance(other, Table):
            return NotImplemented
        return list(self) == list(other) and all(_same(self[name], other[name]) for name in self)

    __hash__ = None

    @property
    def height(self):
        return next(iter(self._columns.values())).size if self._columns else 0

    @property
    def width(self):
        return len(self._columns)

    def row(self, index):
        return {name: _plain(array[index]) for name, array in self._columns.items()}

    def rows(self):
        return [self.row(index) for index in range(self.height)]

    def lookup(self, **criteria):
        mask = np.ones(self.height, dtype=bool)
        for name, value in criteria.items():
            mask &= self._columns[name] == value
        matches = np.flatnonzero(mask)
        if matches.size != 1:
            raise PPIxGPNError("PPIxGPN:InvalidTable", f"Expected one row matching {criteria}, found {matches.size}.")
        return self.row(matches[0])

    def subset(self, selection):
        return Table({name: array[selection] for name, array in self._columns.items()})

    def to_records(self):
        return [{name: _json_value(value) for name, value in record.items()} for record in self.rows()]

    def to_csv(self, path):
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(self._columns)
            for record in self.rows():
                writer.writerow(_csv_value(value) for value in record.values())
        return path

    def __str__(self):
        cells = [[name] + [_display_value(value) for value in array.tolist()] for name, array in self._columns.items()]
        widths = [max(len(cell) for cell in column) for column in cells]
        numeric = [array.dtype.kind in "biuf" for array in self._columns.values()]
        lines = []
        for line in range(self.height + 1):
            parts = [column[line].rjust(width) if is_numeric else column[line].ljust(width)
                     for column, width, is_numeric in zip(cells, widths, numeric)]
            lines.append("  ".join(parts).rstrip())
        return "\n".join(lines)

    def __repr__(self):
        return f"Table({self.height} rows x {self.width} columns)\n{self}"


def _same(first, second):
    if first.shape != second.shape:
        return False
    if first.dtype.kind in "fc" and second.dtype.kind in "fc":
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def _plain(value):
    return value.item() if isinstance(value, np.generic) else value


def _real_text(value, spec):
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    return format(value, spec) if spec else repr(value)


def _csv_value(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return _real_text(value, "")
    return str(value)


def _display_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _real_text(value, ".6g")
    return str(value)


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

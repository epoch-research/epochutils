"""Sync a pandas DataFrame to an Airtable table.

The personal access token needs scopes data.records:read, data.records:write, schema.bases:read and schema.bases:write.

Quick start
-----------
    from epochutils.data.airtable import connect, sync_dataframe

    base = connect(api_key, base_id)
    sync_dataframe(base, "My table", df, "Name", column_types={"Start date": "date"})
"""

import datetime
import math

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from pyairtable import Api, Base, Table


def connect(api_key: str, base_id: str) -> Base:
    return Api(api_key).base(base_id)


def field_spec(name: str, dtype: str, column_type: str | dict | None = None, *,
               precision: int = 8) -> dict:
    """Build an Airtable field definition for a column.

    column_type overrides the dtype inference and may be:
      * a full field dict, e.g. {"type": "singleSelect", "options": {...}}
        (for selects, linked records, anything with options);
      * a bare type string, e.g. "multilineText", "date" ("date" expands
        to an ISO date field);
      * None to infer: numeric dtypes -> number, else singleLineText.
    The "name" key is always added automatically.
    """
    if isinstance(column_type, dict):
        return {"name": name, **column_type}
    if column_type == "date":
        return {
            "name": name,
            "type": "date",
            "options": {"dateFormat": {"name": "iso", "format": "YYYY-MM-DD"}}
        }
    if column_type:
        return {"name": name, "type": column_type}
    if is_numeric_dtype(dtype) and not is_bool_dtype(dtype):
        return {"name": name, "type": "number", "options": {"precision": precision}}
    return {"name": name, "type": "singleLineText"}


def _to_serializable(value):
    """Coerce a cell to something JSON-serializable: numpy scalars to native
    Python, nulls (None/NaN/Inf/NaT) to None, datetimes to ISO strings."""
    if hasattr(value, "item"):
        value = value.item()
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return value


def _canonical_key(value):
    """Canonical form of a primary-key value: integral floats collapse to int,
    so 1, 1.0 and np.int64(1) stay the same key across dtype drift and syncs."""
    value = _to_serializable(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _key_str(value) -> str:
    """String used to compare keys between the source df and Airtable; "" means missing."""
    value = _canonical_key(value)
    return "" if value is None else str(value)


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to Airtable {"fields": {...}} records.

    Nulls (NaN/Inf/None) become explicit None.
    """
    return [{"fields": {k: _to_serializable(v) for k, v in row.to_dict().items()}}
            for _, row in df.iterrows()]


def ensure_table(base: Base, table_name: str, df: pd.DataFrame, primary_field: str, *,
                 column_types: dict | None = None) -> Table:
    """Create table_name if absent, or add any columns it's missing; return the Table.

    column_types maps a column name to a type override (see field_spec):
    a bare type string ("date", "multilineText", ...) or a full field dict
    (for selects, linked records, etc.). Columns not present are inferred from dtype.
    Note Airtable can't change an existing field's type, so an override only takes
    effect when the field is first created.
    """
    column_types = column_types or {}
    columns = [primary_field] + [c for c in df.columns if c != primary_field]
    specs = [field_spec(c, str(df[c].dtype), column_types.get(c)) for c in columns]

    by_name = {t.name: t for t in base.schema().tables}
    if table_name in by_name:
        table_schema = by_name[table_name]
        table = base.table(table_schema.id)
        have = {f.name for f in table_schema.fields}
        for spec in specs:
            if spec["name"] not in have:
                print(f"  + field {spec['name']}")
                table.create_field(spec["name"], spec["type"], options=spec.get("options"))
        return table

    print(f"  creating table '{table_name}'")
    return base.create_table(table_name, fields=specs)


def delete_stale_records(table: Table, df: pd.DataFrame, primary_field: str) -> int:
    """Remove rows whose primary_field value is no longer present in df.

    Upsert never deletes, so without this, rows dropped from the source data
    linger in Airtable forever. Keys are compared canonically (1 == 1.0) so
    numeric dtype drift can't mark live rows stale. Safeguard: if df has no
    keys, skip deletion entirely so a broken pipeline can't wipe the table.
    Returns the count removed.
    """
    new_keys = {_key_str(v) for v in df[primary_field]} - {""}
    if not new_keys:
        print(f"  WARN: empty key set on '{primary_field}', skipping stale-row cleanup")
        return 0
    stale_ids = [
        r["id"] for r in table.all(fields=[primary_field])
        if _key_str(r.get("fields", {}).get(primary_field)) not in new_keys
    ]
    if stale_ids:
        table.batch_delete(stale_ids)
    return len(stale_ids)


def sync_dataframe(base: Base, table_name: str, df: pd.DataFrame, primary_field: str, *,
                   column_types: dict | None = None,
                   prune: bool = True) -> str:
    """Ensure the table exists, upsert every row of df, and prune stale rows.

    Returns the table id. Set prune=False to keep rows that are no longer in df.
    See ensure_table/field_spec for column_types.

    Raises ValueError if any row has a null/empty primary_field: upsert
    can't match such rows, so syncing them would fail mid-batch or create junk.
    """
    bad_rows = [i for i, v in df[primary_field].items() if _key_str(v) == ""]
    if bad_rows:
        raise ValueError(
            f"{len(bad_rows)} row(s) have a null/empty key in '{primary_field}' "
            f"(e.g. index {bad_rows[:5]}); drop or fix them before syncing")
    table = ensure_table(base, table_name, df, primary_field, column_types=column_types)
    records = df_to_records(df)
    for record in records:
        record["fields"][primary_field] = _canonical_key(record["fields"][primary_field])
    table.batch_upsert(records, key_fields=[primary_field], typecast=True)
    if prune:
        delete_stale_records(table, df, primary_field)
    return table.id

"""Tests for the airtable mini-library.

Pure functions (field_spec, _to_serializable, df_to_records) are tested directly.
Record-level orchestration (delete_stale_records) uses pyairtable's
MockAirtable, which intercepts record ops. Schema ops (base.schema /
create_table / create_field) are NOT mocked by MockAirtable, so
ensure_table / sync_dataframe are tested against a tiny fake Base.
"""
import datetime
import json

import numpy as np
import pandas as pd
import pytest
from pyairtable import Api
from pyairtable.testing import MockAirtable

from epochutils.data import airtable


# --------------------------------------------------------------------------- #
# field_spec
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype,expected", [
    ("int64", "number"), ("float64", "number"), ("object", "singleLineText"),
    # pandas nullable dtypes (capitalized) must also infer as number
    ("Int64", "number"), ("Float64", "number"), ("UInt64", "number"),
    # bools are numeric to pandas but should stay text, as before
    ("bool", "singleLineText"), ("boolean", "singleLineText"),
])
def test_field_spec_infers_from_dtype(dtype, expected):
    assert airtable.field_spec("x", dtype)["type"] == expected


def test_field_spec_number_precision():
    assert airtable.field_spec("x", "float64", precision=3)["options"]["precision"] == 3


def test_field_spec_date_shorthand():
    spec = airtable.field_spec("d", "object", "date")
    assert spec["type"] == "date"
    assert spec["options"]["dateFormat"]["name"] == "iso"


def test_field_spec_string_shorthand():
    assert airtable.field_spec("n", "object", "multilineText") == {"name": "n", "type": "multilineText"}


def test_field_spec_dict_passthrough():
    override = {"type": "singleSelect", "options": {"choices": [{"name": "US"}]}}
    assert airtable.field_spec("c", "object", override) == {"name": "c", **override}


def test_field_spec_always_has_name():
    for column_type in (None, "date", "multilineText", {"type": "checkbox"}):
        assert airtable.field_spec("Foo", "object", column_type)["name"] == "Foo"


# --------------------------------------------------------------------------- #
# _to_serializable
# --------------------------------------------------------------------------- #
def test_to_serializable_coerces_numpy_scalars():
    assert airtable._to_serializable(np.int64(3)) == 3 and isinstance(airtable._to_serializable(np.int64(3)), int)
    assert airtable._to_serializable(np.float64(1.5)) == 1.5


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), None])
def test_to_serializable_drops_nan_inf_none(value):
    assert airtable._to_serializable(value) is None


def test_to_serializable_passes_through_plain_values():
    assert airtable._to_serializable("x") == "x"
    assert airtable._to_serializable(["a", "b"]) == ["a", "b"]


def test_to_serializable_converts_datetimes_to_iso_strings():
    assert airtable._to_serializable(pd.Timestamp("2024-01-01")) == "2024-01-01T00:00:00"
    assert airtable._to_serializable(datetime.datetime(2024, 1, 1, 12, 30)) == "2024-01-01T12:30:00"
    assert airtable._to_serializable(datetime.date(2024, 1, 1)) == "2024-01-01"


def test_to_serializable_drops_nat():
    assert airtable._to_serializable(pd.NaT) is None


# --------------------------------------------------------------------------- #
# _key_str / _canonical_key
# --------------------------------------------------------------------------- #
def test_key_str_canonicalizes_numeric_variants():
    # 1, 1.0 and np.int64(1) are the same key regardless of dtype drift
    assert {airtable._key_str(v) for v in (1, 1.0, np.int64(1), np.float64(1.0))} == {"1"}
    assert airtable._key_str(1.5) == "1.5"
    assert airtable._key_str("x") == "x"


@pytest.mark.parametrize("value", [None, float("nan"), pd.NaT, ""])
def test_key_str_maps_missing_to_empty_string(value):
    assert airtable._key_str(value) == ""


# --------------------------------------------------------------------------- #
# df_to_records
# --------------------------------------------------------------------------- #
def test_df_to_records_keeps_nulls_as_none():
    # Explicit None is required so a PATCH upsert clears stale cells in Airtable.
    df = pd.DataFrame({"Name": ["a", "b"], "v": [1.0, float("nan")]})
    assert airtable.df_to_records(df) == [
        {"fields": {"Name": "a", "v": 1.0}},
        {"fields": {"Name": "b", "v": None}},
    ]


def test_df_to_records_keeps_list_values():
    df = pd.DataFrame({"Name": ["a"], "Tags": [["x", "y"]]})
    assert airtable.df_to_records(df)[0]["fields"]["Tags"] == ["x", "y"]


def test_df_to_records_datetime_column_is_json_serializable():
    df = pd.DataFrame({"Name": ["a", "b"],
                       "Start date": pd.to_datetime(["2024-01-01", None])})
    records = airtable.df_to_records(df)
    assert records == [
        {"fields": {"Name": "a", "Start date": "2024-01-01T00:00:00"}},
        {"fields": {"Name": "b", "Start date": None}},  # NaT clears the cell
    ]
    json.dumps(records)  # must not raise


# --------------------------------------------------------------------------- #
# ensure_table / sync_dataframe — tiny fake Base (schema ops aren't mocked)
# --------------------------------------------------------------------------- #
class _FakeField:
    def __init__(self, name):
        self.name = name


class _FakeTableSchema:
    def __init__(self, name, id, fields):
        self.name, self.id = name, id
        self.fields = [_FakeField(f) for f in fields]


class _FakeSchema:
    def __init__(self, tables):
        self.tables = tables


class _FakeTable:
    def __init__(self, id="tblFake"):
        self.id = id
        self.created_fields = []
        self.upserted = None

    def create_field(self, name, field_type, options=None):
        self.created_fields.append((name, field_type))

    def batch_upsert(self, records, key_fields, typecast=False):
        self.upserted = records
        return {}


class _FakeBase:
    def __init__(self, existing=()):
        self._tables = list(existing)
        self.created_tables = []
        self._table = _FakeTable()

    def schema(self):
        return _FakeSchema(self._tables)

    def table(self, id_or_name):
        return self._table

    def create_table(self, name, fields):
        self.created_tables.append((name, [f["name"] for f in fields]))
        return self._table


def test_ensure_table_creates_when_absent():
    base = _FakeBase()
    df = pd.DataFrame({"Name": ["a"], "Value": [1]})
    airtable.ensure_table(base, "T", df, "Name")
    assert base.created_tables == [("T", ["Name", "Value"])]  # primary field first


def test_ensure_table_adds_only_missing_fields():
    base = _FakeBase(existing=[_FakeTableSchema("T", "tbl1", ["Name", "Value"])])
    df = pd.DataFrame({"Name": ["a"], "Value": [1], "Extra": ["z"]})
    airtable.ensure_table(base, "T", df, "Name")
    assert base.created_tables == []                              # not recreated
    assert [n for n, _ in base._table.created_fields] == ["Extra"]  # only the new column


def test_ensure_table_applies_column_types():
    base = _FakeBase(existing=[_FakeTableSchema("T", "tbl1", ["Name"])])
    df = pd.DataFrame({"Name": ["a"], "When": ["2024-01-01"]})
    airtable.ensure_table(base, "T", df, "Name", column_types={"When": "date"})
    assert base._table.created_fields == [("When", "date")]


def test_sync_dataframe_creates_and_upserts():
    base = _FakeBase()
    df = pd.DataFrame({"Name": ["a"], "V": [1]})
    airtable.sync_dataframe(base, "T", df, "Name", prune=False)
    assert base.created_tables[0][0] == "T"
    assert base._table.upserted == [{"fields": {"Name": "a", "V": 1}}]


def test_sync_dataframe_canonicalizes_key_in_upsert_payload():
    # Integral float keys go out as ints so a text primary field never stores "1.0".
    base = _FakeBase()
    df = pd.DataFrame({"ID": [1.0], "V": [2.5]})
    airtable.sync_dataframe(base, "T", df, "ID", prune=False)
    key = base._table.upserted[0]["fields"]["ID"]
    assert key == 1 and isinstance(key, int)  # not float 1.0 (1 == 1.0 in Python)


def test_sync_dataframe_sends_none_for_null_cells():
    base = _FakeBase()
    df = pd.DataFrame({"Name": ["a"], "V": [float("nan")]})
    airtable.sync_dataframe(base, "T", df, "Name", prune=False)
    assert base._table.upserted == [{"fields": {"Name": "a", "V": None}}]


@pytest.mark.parametrize("bad_key", [None, float("nan"), ""])
def test_sync_dataframe_rejects_null_or_empty_keys(bad_key):
    base = _FakeBase()
    df = pd.DataFrame({"Name": ["a", bad_key], "V": [1, 2]})
    with pytest.raises(ValueError, match="null/empty key"):
        airtable.sync_dataframe(base, "T", df, "Name", prune=False)
    assert base._table.upserted is None  # failed before any network call


# --------------------------------------------------------------------------- #
# delete_stale_records — MockAirtable covers list/delete
# --------------------------------------------------------------------------- #
def test_delete_stale_removes_absent_keys():
    with MockAirtable() as mock:
        table = Api("patFAKE").base("appFAKE").table("tblFAKE")
        mock.add_records(table, [{"fields": {"Name": n}} for n in ("a", "b", "c")])
        removed = airtable.delete_stale_records(table, pd.DataFrame({"Name": ["a", "c"]}), "Name")
        assert removed == 1
        assert sorted(r["fields"]["Name"] for r in table.all()) == ["a", "c"]


def test_delete_stale_keeps_numeric_keys_across_dtype_drift():
    # Regression: int keys stored in Airtable vs float64 source column ("1" vs
    # "1.0") used to mark every live row stale and wipe the table.
    with MockAirtable() as mock:
        table = Api("patFAKE").base("appFAKE").table("tblFAKE")
        mock.add_records(table, [{"fields": {"ID": 1}}, {"fields": {"ID": 2}}])
        removed = airtable.delete_stale_records(table, pd.DataFrame({"ID": [1.0, 2.0]}), "ID")
        assert removed == 0
        assert len(table.all()) == 2


def test_delete_stale_empty_key_set_is_a_noop():
    with MockAirtable() as mock:
        table = Api("patFAKE").base("appFAKE").table("tblFAKE")
        mock.add_records(table, [{"fields": {"Name": n}} for n in ("a", "b")])
        removed = airtable.delete_stale_records(table, pd.DataFrame({"Name": [None, ""]}), "Name")
        assert removed == 0
        assert len(table.all()) == 2  # safeguard: nothing deleted

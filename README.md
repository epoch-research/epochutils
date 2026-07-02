# [PUBLIC] epochutils
Bag of utilities for internal use

## epochutils.data.airtable

Sync a pandas DataFrame to an Airtable table.

```python
from epochutils.data.airtable import connect, sync_dataframe

base = connect(api_key, base_id)
sync_dataframe(base, "My table", df, "Name", column_types={"Start date": "date"})
```

To set up your data repo to upload the data to Airtable via GitHub Actions, point your agent to [docs/airtable-upload-setup.md](docs/airtable-upload-setup.md).

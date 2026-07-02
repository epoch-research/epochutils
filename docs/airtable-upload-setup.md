# Setting up Airtable upload for a data repo (for AI agents)

How to give a data repo a GitHub Actions workflow that uploads its generated tables to Airtable — **prod on push to `main`, a test base on manual dispatch** — using the [`epochutils`](../README.md) library.

Work through the steps in order. Legend:

- 🤖 **you (the agent) do this** — run the commands / write the files.
- 🙋 **ask the user** — a manual step that can't be automated (Airtable UI, token creation, etc.). Do it by pausing and asking.

> Read the whole guide before starting, then confirm the plan with the user. Several Airtable operations (creating bases, creating tokens, deleting tables) have **no API** and must be done by a human — plan for those hand-offs.

---

## Prerequisites

- The repo **already has working data-generation code** that yields the tables you want to upload. One of:
  - an importable function returning DataFrames (e.g. `get_all_tables()` → `{name: DataFrame}`), or
  - notebooks/scripts that write CSVs to a known directory.
- `gh` CLI authenticated for the org (`gh auth status`), and `uv` installed. 🙋 If `gh` isn't installed or authenticated, ask the user whether to set it up or to do the GitHub steps in the web UI themselves.
- Read access to the `epochutils` library repo (this repo).

---

## What you'll end up with

```
<data-repo>/
├── upload_to_airtable.py           # calls generation, then epochutils.sync_dataframe
├── pyproject.toml                  # + epochutils git dependency
├── uv.lock                         # pins epochutils to a commit
└── .github/
    ├── workflows/upload-to-airtable.yml   # copied from this repo's docs/
    └── CODEOWNERS
```

Plus, on GitHub: two **Environments** (`production`, `test`), their base-id variables and per-environment Airtable token secrets, and branch protection on `main`.

---

## Step 1 🤖 — Write `upload_to_airtable.py`

The script (a) **calls the data generation** — never reads a hand-committed CSV as the source of truth — and (b) uploads each table with `epochutils.data.airtable.sync_dataframe`.

**Pattern A — importable generator:**

```python
import os
from epochutils.data import airtable

def main():
    api_key = os.environ["AIRTABLE_API_KEY"]
    base_id = os.environ["AIRTABLE_BASE_ID"]
    from my_generator import get_all_tables          # your repo's generation entry point
    tables = get_all_tables()                         # {key: DataFrame}
    base = airtable.connect(api_key, base_id)
    for key, (table_name, primary) in TABLE_CONFIG.items():
        airtable.sync_dataframe(base, table_name, tables[key], primary,
                                column_types=COLUMN_TYPES)
```

**Pattern B — notebook generation:** run the notebooks first, then read what they wrote. Use `papermill`; force the kernel and a headless matplotlib backend (see gotchas):

```python
import papermill as pm, os
os.environ.setdefault("MPLBACKEND", "Agg")
for nb in NOTEBOOKS:
    pm.execute_notebook(nb, f"/tmp/{nb}", kernel_name="python3", cwd=".", progress_bar=False)
# then read the generated CSVs and sync_dataframe(...) each one
```
### `column_types` — get every field right the first time

`sync_dataframe(..., column_types={col: <type>})` overrides per-column field types. **Airtable cannot change a field's type after creation** (see gotchas), so specify anything non-default here:

| Column kind | `column_types` value |
|---|---|
| Date | `"date"` |
| Long text / URLs | `"multilineText"` |
| Single select | `{"type": "singleSelect", "options": {"choices": [{"name": "A"}, …]}}` |
| Multi select | `{"type": "multipleSelects", "options": {"choices": […]}}` |
| Currency | `{"type": "currency", "options": {"precision": 2, "symbol": "$"}}` |
| Checkbox | `{"type": "checkbox", "options": {"icon": "check", "color": "greenBright"}}` |
| Link to another table | `{"type": "multipleRecordLinks", "options": {"linkedTableId": "<tblId>"}}` |

Columns you don't list are inferred: numeric → `number`, everything else → `singleLineText`.

### Value shaping before upload

- **Primary/merge key** must be unique and non-null in every row.
- **Dates:** parse string columns with `pd.to_datetime(...)` — source data often mixes `M/D/YYYY` and ISO, so don't rely on Airtable to guess. `datetime64` columns are serialized to ISO strings automatically; no manual `strftime` needed.
- **Currency:** strip `$`/`,` → float.
- **Checkbox:** map to `True` / `None` (omit = unchecked).
- **Multi-select:** pass a list of option strings; split comma-joined source values.
- **Linked records:** the cell must be a **list of the linked table's primary-field values**, e.g. `"AMD"` → `["AMD"]`. Sync the linked (dimension) table **first**, and make sure the values **exactly match** its primary field (normalize casing/aliases, e.g. `"NVIDIA"`→`"Nvidia"`, `"Amazon AWS"`→`"Amazon"`) or Airtable will create duplicate rows.

`sync_dataframe` upserts on the primary key (idempotent re-runs) and prunes rows no longer present; it raises up front if any row's key is null or empty, so a broken pipeline fails before touching the table.

---

## Step 2 🤖 — Add the `epochutils` dependency

In the data repo's `pyproject.toml`:

```toml
dependencies = ["pandas>=2.0", "epochutils", ...]   # + papermill, ipykernel, squigglepy, matplotlib for notebook repos

[tool.uv.sources]
epochutils = { git = "https://github.com/epoch-research/epochutils.git", branch = "main" }
```

Then `uv lock` and commit `uv.lock` (pins the exact `epochutils` commit + the whole dependency tree).

---

## Step 3 🙋 — Create the Airtable bases

Ask the user to create **two bases** and share their `app…` IDs:

- **production** — name it plainly, e.g. `Chip Sales`.
- **test** — same name + `(test)`, e.g. `Chip Sales (test)`.

> Why manual: the Airtable "create base" API requires a **workspace ID** that isn't discoverable through the standard API, so an agent can't reliably create a base. (If the user gives you a workspace ID you *can* create bases with `pyairtable`'s `Api.create_base` — but confirm first.)

Record which ID is prod and which is test — misrouting prod is the costliest mistake here.

---

## Step 4 🙋 — Create the Airtable tokens

Ask the user to create **two personal access tokens** at <https://airtable.com/create/tokens>, each with scopes `data.records:read`, `data.records:write`, `schema.bases:read`, `schema.bases:write`: one with access to the **production base only**, one to the **test base only**.

> Why manual: token creation is a browser-only flow; an agent can't mint a PAT.

**Never have the user paste a token to you** — anything sent to the agent can persist in chat, session, or tool logs. The user stores the token themselves via an interactive `gh secret set` (Step 5), whose hidden prompt keeps the token bytes out of agent context entirely.

---

## Step 5 🤖 + 🙋 — Configure GitHub environments, variables, secrets

Two environments; `production` restricted to `main`:

```bash
R=epoch-research/<data-repo>
# environments
gh api -X PUT repos/$R/environments/test --silent
gh api -X PUT repos/$R/environments/production \
  -F "deployment_branch_policy[protected_branches]=false" \
  -F "deployment_branch_policy[custom_branch_policies]=true" --silent
gh api -X POST repos/$R/environments/production/deployment-branch-policies -f name=main --silent

# base ids as ENV VARIABLES (not secret — they're in the base URL)
gh variable set AIRTABLE_BASE_ID --env test       --repo $R --body appTESTxxxxxxxxxx
gh variable set AIRTABLE_BASE_ID --env production  --repo $R --body appPRODxxxxxxxxxx
```

> If the test token ever must have more than test-base access, gate the `test` environment with a required reviewer (deployment protection rules apply to any job targeting the environment). With a test-only token, that friction buys little.

🙋 The tokens are stored as ENV SECRETS **by the user, not the agent** — ask them to run these in their own terminal (in Claude Code, `! <command>` runs it in-session):

```bash
gh secret set AIRTABLE_API_KEY --env test        --repo epoch-research/<data-repo>
gh secret set AIRTABLE_API_KEY --env production  --repo epoch-research/<data-repo>
```

With no `--body` and no piped stdin, `gh secret set` opens a hidden prompt to paste the token into — it never enters agent context, argv, or shell history.

---

## Step 6 🤖 — Add the workflow

Copy [`upload-to-airtable.yml`](upload-to-airtable.yml) from this repo into the data repo's `.github/workflows/`. **Delete** the `MPLBACKEND: Agg` line if the upload runs no matplotlib (keep it for the notebook path) — remove it, don't leave it commented out. It routes push→`main` to the `production` environment and manual dispatch to `test`.

---

## Step 7 🤖 — Branch protection + CODEOWNERS

Branch protection on `main` is **required** — it's the real security boundary (see gotchas). Add a `CODEOWNERS` (e.g. `/.github/ @epoch-research/webdev-team`; the team needs push access for it to count) and protect `main`:

```bash
gh api -X PUT repos/$R/branches/main/protection --input - <<'JSON'
{
  "required_status_checks": null,
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "require_code_owner_reviews": true,
    "dismiss_stale_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
JSON
```

---

## Step 8 🙋 — Review table & column names with the user *before* the first upload

**Do not run the script until the user has signed off on the names.** Airtable field and table names are effectively permanent — types are immutable and there's **no API to delete a table or field** (see gotchas) — so a wrong or misspelled name isn't a cheap fix; it means manual UI cleanup. Catch it before anything is created, not after.

Present a compact, per-table summary for the user to check, and pause for explicit confirmation:

- **Table names** (exactly as they'll appear) and each table's **primary field**.
- **Every column → its field type**, calling out the non-default ones — dates, selects, currency, checkboxes, **links** (and the table each links to), and **formulas** (with their exact text).
- Which fields are **created by the script** vs. which the user must **add by hand** (lookups/rollups, and any formula that references one — see the matching section).

```
Table: Quarterly by chip   (primary: Name)
  Name .................. singleLineText
  Chip manufacturer ..... link → Organization
  Start date ............ date
  Number of units (median) .. number
  ...
```

If you're **matching an existing base**, dry-validate every planned column name against that base's real field names first (see the matching section) and show the user the diff — don't just show your intended names.

Ask the user to confirm the names/types are right (and, for an existing base, that nothing is missing or misspelled). Only proceed once they've said so.

## Step 9 — First run, verify, and clean up

1. 🤖 Trigger a **manual dispatch** (goes to the **test** base): `gh workflow run upload-to-airtable.yml --repo $R`. Watch it: `gh run watch <id> --repo $R --exit-status`.
2. 🤖 Verify the test base has the expected tables/rows (via `pyairtable`).
3. 🙋 **Ask the user to delete the default `Table 1`** in the test base. Airtable auto-creates a `Table 1` in every new base and there's **no API to delete a table** — so this is a manual UI step (right-click the table tab → *Delete table*).
4. 🤖 Push to `main` (or merge a PR) → uploads to **prod**. Verify.
5. 🙋 Ask the user to delete `Table 1` in the **prod** base too.

---

## Gotchas & lessons (read before you start)

- **Airtable field types are immutable.** You cannot change a field's type via the API after creation — so `column_types` must be right the *first* time. If a table already exists with the wrong type, the only fix is to **delete and recreate the table** (manual — see next point).
- **No API to delete tables or bases.** `Table 1` cleanup, and fixing a wrong-schema table, are manual UI steps. Only *records* and *fields-you-add* are API-manageable (and fields can't be deleted either — only added/renamed).
- **Creating a base** needs a workspace ID not exposed by the API → manual.
- **Creating a PAT** is browser-only → manual.
- **Linked records:** cell value = list of the target's primary-field values; sync the dimension table first; values must match exactly (normalize aliases/casing) or you get duplicate linked rows. A linked field also auto-creates a reverse field in the target table.
- **Notebooks:** they may pin a kernel that doesn't exist on the runner (e.g. `plotly_kernel`) — force `kernel_name="python3"` in papermill. Set `MPLBACKEND=Agg` for headless matplotlib. Add `papermill` + `ipykernel` (and the notebooks' own deps) to `pyproject.toml`.
- **Security boundary = the environment, not the workflow `if:`.** A `workflow_dispatch` can run branch-edited workflow code, so trigger conditions aren't trustworthy. Prod is safe because the `production` environment is restricted to `main`. **Branch protection on `main` is required** for this to hold (otherwise anyone with write access pushes straight to prod, and `CODEOWNERS` is inert). Ideally also use separate prod/test tokens so the unrestricted `test` environment never holds a prod-capable token.
- **Pin actions to commit SHAs** and stay off deprecated Node versions (checkout `v7`, setup-uv `v8`).
- **`typecast=True`** lets Airtable coerce values on upsert, but still normalize dates/currency yourself — don't depend on it for correctness.
- **Re-runs are safe:** `sync_dataframe` upserts on the primary key and prunes stale rows (raising up front if any key is null or empty).

---

## Matching an existing base's exact schema

If the goal is to *reproduce an existing* Airtable base (not just upload fresh tables), lessons from doing it:

- **Pull the target schema from the real base first.** `GET /meta/bases/{id}/tables` exposes each field's type, link `linkedTableId`, lookup config (`recordLinkFieldId` + `fieldIdInLinkedTable`), and **formula text** (with `{fld…}` refs — translate those IDs to field names to reuse the formula). Save it as a blueprint and build from it.
- **Lookups and rollups can't be created via the API or MCP** — only scalar / select / date / checkbox / currency / **link** / **formula**. Build the scalar + link + (lookup-free) formula fields programmatically; add lookups — and any formula that references one — by hand in the UI. Generate a "manual fields" checklist from the blueprint.
- **Match each table's field names exactly.** Real bases drift: the same concept can be `Number of units (median)` in one table and `Number of Units` in another; `Compute estimate in H100e (…)` vs `H100e compute power (…)`. Use a per-table rename map to the real names.
- **Source files are often inconsistent across each other.** One designer's CSVs used different column names and carried extra columns. Normalize the variants, **coalesce duplicate-named columns** (first non-null per row — keep it dtype-preserving; a transpose-groupby stringifies numbers), then **select only the real field set** so no stray column silently creates a junk field.
- **Dry-validate before running.** Fields and tables are permanent (no delete API), so check that every built column name is ∈ the real table's field names *before* the first upload — it catches rename/normalization mistakes while they're still free to fix.
- **Link ordering:** create/populate the linked-to (dimension) table first; link cell values are arrays of its primary-field values, matched case-sensitively (normalize aliases like `NVIDIA`→`Nvidia`).
- **Reaching a brand-new base:** a PAT scoped to "all current and future bases" can write to a just-created base; otherwise create and populate it through the MCP (OAuth). Creating a base via the plain REST API needs a workspace ID.

---

## Found a problem?

If anything here was wrong, missing, or behaved unexpectedly during setup, then at the end of the process 🙋 **ask the user whether to update this guide** — ideally by opening a PR against `epoch-research/epochutils` with the fix, so the next agent has it easier.

# R1 documentation truth audit — 2026-07-23

## Scope and base

- Branch: `Claude`
- Base commit: `cd7d37b` (`docs: close R1 TODO reconciliation`)
- Scope: README, Hugo source pages, `docs/api-contract-2.0.md`, CHANGELOG,
  and documentation-bearing source/build comments.
- Constraint: non-behavioral documentation/record changes only.

## Preregistered acceptance band

1. Every corrected load-bearing claim names current code, an executable result,
   a committed baseline, or a verified citation. Unsupported prediction is
   removed or explicitly tagged inferred.
2. Known stale claims from the TODO reconciliation are resolved:
   Float64 default; live metric-dispatch home; removed SIMD surface; accurate
   UCR/kernel performance ranges; current FastCLARA and F7 behavior.
3. Rebuild `build/highs-1151`, then run the real CLI documentation gate:

   ```text
   python scripts/check_docs_contract.py --cli build/highs-1151/bin/dtwc_cl.exe
   ```

   Decisive band: exit 0 and exact terminal line
   `documentation contract checks passed`. Generated-source drift, frozen
   contract governance, migration behavior, rc1 CHANGELOG structure, HPC error
   messages, Tier-1 signatures/method registries, live CLI flags, and Float64
   CLI documentation are all in this gate.
4. Run `scripts/check_site_links.py` only against a site freshly generated from
   the edited sources. Decisive band: exit 0 and exact line
   `all internal site links resolve`. If Hugo cannot run locally, record
   `[BLOCKED-ENV]` with the tool probe verbatim. A pass over the pre-existing
   ignored `docs/public/` tree is advisory and cannot close the fresh-site gate.
5. `git diff --check` emits no errors. The final diff contains no program
   behavior changes.

## Environment probe

```text
hugo=NOT_FOUND
go=NOT_FOUND
node=C:\Program Files\nodejs\node.exe
v24.16.0
npx=C:\Program Files\nodejs\npx.ps1
11.13.0
cli_exists=True
tracked_public=0
!! docs/public/
```

**[BLOCKED-ENV]** A fresh Hugo site and therefore the decisive internal-link
gate cannot be produced on this host: both `hugo` and its `go` build fallback
are absent. Continue with source-level audit, the live CLI contract gate, and
an explicitly advisory check of the existing ignored site. Hosted CI remains
operator-owned and is not claimed.

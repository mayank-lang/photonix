# Putting photonix on GitHub

The repository is GitHub-ready: packaging (`pyproject.toml`), `LICENSE` (MIT),
`README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `.gitignore`, and CI
(`.github/workflows/ci.yml`) are all in place, and `ruff` + the test suite pass.

Initialize and push **from your own machine** (git can't run through this
session's mounted folder). In PowerShell, from the `photonix/` directory:

```powershell
# 0. Remove the empty .git stub this session created (it could not finish here)
Remove-Item -Recurse -Force .git -ErrorAction SilentlyContinue

# 1. Initialize and make the first commit
git init
git add -A
git commit -m "photonix v0.1: differentiable PIC design + MEEP FDTD backend"

# 2. Create the repo on GitHub (either via the website, or the GitHub CLI):
#    gh repo create photonix --public --source . --remote origin --push
#
#    Or, if you made the empty repo on github.com manually:
git branch -M main
git remote add origin https://github.com/<your-username>/photonix.git
git push -u origin main
```

## Before you push — a couple of optional cleanups

- **Update project URLs.** `pyproject.toml` and `README.md` use the placeholder
  `github.com/photonix/photonix`. Replace `photonix/photonix` with
  `<your-username>/photonix`.
- **Author metadata.** `pyproject.toml` lists `authors = [{ name = "photonix
  contributors" }]` — set your name/email if you want.

## What CI will run (GitHub Actions)

On every push/PR to `main`, `.github/workflows/ci.yml`:
1. installs `.[dev]` on Python 3.10 / 3.11 / 3.12,
2. runs `ruff check src tests`,
3. runs `pytest tests -q` (the MEEP-backend tests skip automatically — MEEP is
   conda-only and not installed in CI),
4. runs the photonix-only benchmark as a smoke test.

MEEP itself is intentionally **not** a CI dependency; the backend is validated
locally on a MEEP-equipped machine (see `MEEP_BACKEND_RUN.md`).

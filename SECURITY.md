# Security policy

## Supported versions

photonix is pre-1.0, so only the latest minor release line gets fixes. Security
fixes are released as a new patch version of that line.

| Version | Supported |
|---|---|
| 0.2.x | yes |
| < 0.2 | no (never published) |

## Reporting a vulnerability

Please **do not open a public issue** for a suspected vulnerability. Report it
privately through GitHub's private vulnerability reporting: go to
<https://github.com/mayank-lang/photonix/security/advisories/new>, or open the
repository's **Security** tab and choose **Report a vulnerability**.

Include the affected version (`photonix info --json`), a minimal reproduction,
and the impact you expect. You should get an acknowledgement within 7 days. A fix
or mitigation, and a coordinated disclosure date, follow once the report is
confirmed.

## Scope

photonix is a library and command-line tool that reads user-supplied files and
can start external programs. These are in scope:

- Parsing untrusted input: Touchstone (`.sNp`), `SParameterDataset` NPZ (loaded
  with `allow_pickle=False`), netlist JSON, GDSII/OASIS (through gdstk) and the
  CLI's rules files.
- Starting external tools: the KLayout DRC/LVS runner and the Elmer, DEVSIM and
  Lumerical adapters build argument vectors and use `shell=False`. Any way to
  inject shell commands or arguments is in scope.
- The release pipeline (`.github/workflows/release.yml`), which publishes to
  PyPI through OIDC Trusted Publishing with no stored tokens.

Out of scope: the numerical accuracy of a model. That is a correctness issue, so
report it with the *validation discrepancy* issue template. Also out of scope:
vulnerabilities in optional third-party solvers (Meep, KLayout, Lumerical); report
those upstream.

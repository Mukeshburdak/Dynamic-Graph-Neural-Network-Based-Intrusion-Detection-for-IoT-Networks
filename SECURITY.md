# Security Policy

## Supported Versions

This is an academic/research project without formal versioned releases.
Only the latest code on the `main` branch is maintained and supported with
fixes.

| Version         | Supported          |
| ---------------- | ------------------ |
| `main` (latest)  | :white_check_mark: |
| Older commits    | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this project (for example, in
the Flask dashboard's `/predict` endpoint, dependency vulnerabilities, or
unsafe handling of uploaded/input data), please report it responsibly:

- **Preferred:** open a [GitHub Security Advisory](../../security/advisories/new)
  on this repository (private, visible only to maintainers until resolved).
- **Alternative:** open a regular [GitHub Issue](../../issues), but avoid
  posting exploit details publicly — describe the issue at a high level and
  a maintainer will follow up to get specifics privately.

Please include:
- A description of the vulnerability and its potential impact
- Steps to reproduce it
- Any suggested fix, if you have one

**What to expect:** as this is a student/research project maintained
part-time, please allow a reasonable window for a response rather than
expecting immediate action. Valid reports will be acknowledged, and a fix
or mitigation will be prioritized based on severity.

## Scope Note

This project is a research/educational pipeline (data preprocessing, graph
construction, GNN training, and a local demo dashboard) and is **not**
intended for production deployment or use with live/sensitive network
traffic without further security hardening (e.g. the dashboard's
`/predict` endpoint has no authentication and is intended for local,
single-user demo use only).

# Security Policy

## Supported Versions

PixelFlux is a prototype. Security fixes are made on `main` and included in the next build. There
are no supported older versions.

## Reporting a Vulnerability

Please report suspected vulnerabilities through GitHub private vulnerability reporting:

https://github.com/monstercameron/PixelFlux/security/advisories/new

Do not open a public issue for an unpatched vulnerability. Include the affected component, the
steps to reproduce, the impact, and any relevant logs with personal paths removed.

## Handling

Reports are triaged as soon as practical. Confirmed issues that cause PixelFlux to make a network
request it did not announce, to write outside its own data directory, or to expose the contents of
a library to another process are treated as blocking until fixed or explicitly deferred with a
documented mitigation.

## Scope Notes

PixelFlux is a local desktop application. Its security story is mostly about what it does *not*
do, so those are the claims worth attacking:

- **One network call, ever.** The first-run model download is the only request the application is
  designed to make, and it happens because you pressed a button. Browsing, analysis and search make
  none, and the interface's content security policy forbids them. A reproducible request outside
  that one download is a vulnerability, not a bug.
- **Your library stays where it is.** Photographs are read in place and indexed into a local SQLite
  file. Nothing is uploaded and nothing is copied outside the data directory.
- **Faces are appearance, not identity.** Grouping is recomputed on every page load and never
  stored; only names you type are persisted. Treat any behaviour that persists a face grouping as
  a privacy defect.

**Never attach real photographs, a real library database, or the contents of your data directory
to a report.** The test corpus is enough to reproduce almost anything.

## Out of Scope

- The models themselves. They are third-party weights downloaded on request; their outputs are
  treated as search material and are documented as sometimes wrong. A model describing a photograph
  incorrectly is a known limitation, not a vulnerability.
- Anything requiring an attacker who already has your user account on the machine. PixelFlux runs
  as you, with your files. It is not a sandbox and does not claim to be one.

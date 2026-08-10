# What this changes

<!-- One or two sentences. The diff says what the code does; say why it does it. -->

## Why

<!-- What was wrong, or what this makes possible. If it reverses an earlier decision, say what
     new evidence reversed it — the comments in this repository record reasons, not mechanics. -->

## Measurements

<!-- Required for any performance or accuracy claim. Include the numbers AND the command that
     produced them. Nearly every interesting decision here came from a measurement contradicting
     the obvious guess; "it feels faster" is not one. Delete this section if it does not apply. -->

## Checks

- [ ] `dotnet build` is clean
- [ ] `dotnet test` passes — note which filter you used if you do not have the models or the corpus
- [ ] If this changes how a search vector is produced (a blend weight, a preprocessing step, a model
      swap), **the model version string changed with it**. Otherwise old vectors are silently reused
      and the change looks like it did nothing. That has already happened once.
- [ ] No new network call. Browsing, analysis and search make none, and the CSP forbids them.
- [ ] No real photographs, library databases, or personal paths in the diff, the tests, or the
      screenshots.

## Anything a reviewer should push back on

<!-- Shortcuts taken, things you were unsure about, parts you would like a second opinion on. -->

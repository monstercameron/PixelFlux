# Contributing to PixelFlux

PixelFlux is a prototype built in spare time. Contributions are welcome, and so is the honest
warning that the architecture is still moving.

Before anything else, read the **Honest limitations** section of the [README](README.md). Several
things that look like bugs are documented decisions — the vision model inventing detail, face
grouping being appearance rather than identity, Windows being the only tested platform.

## Before you open a pull request

Open an issue first for anything larger than a fix. The measurements in the README are what this
project is *for*, and a change that improves one number at the cost of another needs to be a
conversation rather than a surprise in a diff.

## Building

Windows, .NET 10, MAUI Blazor Hybrid. arm64 is the tested target; x64 builds and passes but gets
far less use.

```
dotnet build
dotnet test
```

`dotnet test` runs 207 tests. Most are pure logic and run anywhere, but **roughly a third do real
inference against the real models and ten need the photograph corpus** — and on a fresh clone
those *fail* rather than skip. Fetch the models and the corpus first (`tools/fetch_runtime.py`,
`tools/fetch_test_album.py`), or filter to the parts that do not need them:

```
dotnet test --filter "FullyQualifiedName~PipelineTests|FullyQualifiedName~PersonSegmentTests|FullyQualifiedName~PeopleTests|FullyQualifiedName~Localisation"
```

Making the model-dependent tests skip cleanly instead of failing is worth doing and has not been
done. It is a good first contribution.

## What a good change looks like

**Measure it.** Nearly every interesting decision in this project came from a measurement
contradicting the obvious guess: a caption in the search vector beat a better image encoder, the
bottleneck was JPEG decode rather than the model, and a graphics processor turned out not to be
uniformly faster. If your change is a performance or accuracy claim, include the numbers and the
command that produced them.

**Write the reason down, in the code.** The comments in this repository record why a thing is the
way it is, not what the line does. If you reverse a decision, say what new evidence reversed it.

**Do not add a network call.** Browsing, analysis and search make no requests, the content security
policy forbids them, and the first-run model download is the single announced exception. That
property is load-bearing and a change that quietly breaks it will be reverted.

**Keep the model version string honest.** Anything that changes how a vector is produced — a
blend weight, a preprocessing step, a model swap — must change the version string with it.
Otherwise old vectors are silently reused and the change looks like it did nothing. That has
already happened once.

## Reporting bugs and vulnerabilities

Ordinary bugs: open an issue with the platform, the build, and the steps.

Security or privacy issues — anything where PixelFlux makes a request it should not, writes outside
its data directory, or exposes a library — go through
[SECURITY.md](SECURITY.md), not a public issue.

**Never attach real photographs or a real library database** to an issue or a pull request. The
test corpus reproduces almost everything.

## Code of Conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

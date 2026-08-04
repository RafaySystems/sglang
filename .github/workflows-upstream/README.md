# Upstream workflows — parked, not deleted

GitHub only runs workflows from `.github/workflows/`, so the 62 files here are
disabled in this fork.

## What forced it

Seven upstream workflows fire on `push.tags: 'v[0-9]+.*'`:

```
bot-bump-docs-version   release-docker          release-docker-amd
release-docker-npu      release-docker-runtime  release-docker-xeon
release-pypi
```

In GitHub's filter syntax `+` means "one or more of the preceding character",
so that pattern reads *v, one or more digits, a dot, anything* — and
`v0.5.16-rafay.1` matches it. Pushing a Rafay tag would start upstream's docker
releases and a **PyPI publish**, against a build that is not theirs.

The remaining 55 were parked because they fire on `push` to branches or on
`schedule`: nightly GPU suites, AMD/NPU/XPU/MUSA test matrices, link checkers,
stale bots. On a fork they burn minutes and fail against infrastructure they
were never pointed at.

## What was kept

32 files that declare only `workflow_call` and/or `workflow_dispatch`. **These
cannot fire on their own** — they run when another workflow calls them, or when
a human asks. Keeping them active costs nothing and preserves the reusable
building blocks.

## Why moved rather than deleted

A rename keeps upstream's edits mergeable: a change to `release-pypi.yml`
applies to the file at its new path instead of conflicting as
modified-by-them/deleted-by-us on every rebase — 62 such conflicts, every time.

## The gap

A workflow **added** upstream lands in `.github/workflows/` and is live here
immediately. With 94 files upstream and a fast-moving project, this is worth a
glance after each rebase.

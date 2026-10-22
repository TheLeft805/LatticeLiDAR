# Security scope and current limitations

LatticeLiDAR is a local research codec, not a network service or an AI agent.
Treat point-cloud files, compressed streams, model files and third-party code as
separate trust boundaries. Do not expose this frozen release to arbitrary uploads.

## Checkpoints: mitigation, not a library fix

The frozen environment uses PyTorch 2.2.1+cu118. It is in the affected ranges of
[CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)
and [CVE-2026-24747](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
`weights_only=True` alone does not make malicious checkpoints safe. The latter
advisory lists 2.10.0 and later as patched for that issue; this is not a claim
that a new PyTorch version works with this repository's frozen CUDA stack.

`model_config.create_codec` (the release CLIs) and the GPU regression entry point
check the checkpoint and reconstruction archive against the trusted `ASSETS.json`
allowlist **before** importing/constructing the model. Byte-identical copies at
other paths are accepted; modified/custom assets are rejected. There is no CLI
flag to bypass this check. Do not edit the trusted manifest to accept an unknown
checkpoint. Keep the checkout and assets protected from concurrent untrusted
writes: a preflight hash check is not protection against local file replacement
between verification and loading.

Direct research use of `GeometryCodec` or `ReconstructionCodec` does not go
through that preflight. Those APIs must only receive trusted files. The allowlist
does not repair the underlying dependency vulnerability or establish authenticity
if the repository and manifest themselves are compromised.

A supported patched dependency stack or a non-pickle weight migration requires
real GPU, numerical and bitstream compatibility tests before adoption. Do not
silently upgrade a single dependency and assume interoperability.

## Streams, native execution and resource limits

Existing CRC, identity, state and extent checks detect malformed input but are
not sender authentication. Never treat a valid CRC as proof of trust. CLI size
budgets do not bound every downstream CPU/GPU allocation. The direct decoder API
and ANS workspace amplification still need profile-aware allocation-budget review.

Native address-width changes and malformed-input `compute-sanitizer` coverage
remain follow-up work. A signed `lane * capacity` calculation was identified as
a static boundary risk; no exploitable out-of-bounds access was reproduced.
Ordinary corruption rejected by the outer CRC does not test all native paths.

JIT compilation loads native code from the local cache. Keep the checkout,
compiler search paths and cache private to trusted users. Do not run unreviewed
contributions or automated tests with production credentials.

## Reporting

For a suspected vulnerability, use GitHub private vulnerability reporting if the
repository offers it, or contact the maintainer privately through a verified
channel. Do not publish exploit payloads or secrets in a public issue. No security
response time or fully completed audit is claimed by this release.

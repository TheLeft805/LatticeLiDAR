# Frozen codec compatibility

Tooling maintenance preserves the numerical source and model assets recorded in
`ASSETS.json` for the 0.1.1 snapshot. Run `python tools/verify_assets.py` to check
that invariant. This is a static integrity check, not a real GPU roundtrip.

The stream identity includes source bytes, weight bytes, the profile,
quantization and PyTorch/CUDA versions. Reconstruction also binds its source and
head assets. Identical entropy bodies alone do not imply complete-stream
compatibility. Even a non-numerical edit inside a hashed source changes identity.
The historical 0.1.0 and 0.1.1 implementations therefore must not be assumed to
interoperate. The strict decoder rejection is intentional and must not be removed
just to make an old stream load.

For archived data, retain the exact trusted source revision, weights, profile and
environment specification used to produce it. Use old environments only for
trusted files and isolate them: preserving an environment does not make vulnerable
dependencies safe. There is no automatic cross-version migration in this release.

Before changing the numerical implementation or dependencies:

1. Keep trusted golden streams and their exact decoded references from the old
   revision, with checksums and environment metadata.
2. Run independent encoder/decoder processes at all four quantization scales.
3. Check integer geometry, reconstruction, probability traces, repeated-frame
   determinism and rejection/recovery behavior.
4. Explicitly record compatible and incompatible version pairs. If migration is
   needed, decode with the correct trusted old environment and re-encode under the
   new version; verify the intended geometry/quality semantics.
5. Update release manifests only after validation. A matching source hash is a
   guardrail, not a substitute for numerical testing across hardware/toolchains.

This maintenance change does not migrate PyTorch, change CUDA kernels or remove
identity checks. Supported GPU scope remains sm_89; other GPUs are not validated.

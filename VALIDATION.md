# Validation and maintenance

## CPU checks (not a GPU codec pass)

From the repository root, with Python 3.10 and NumPy 1.26.4:

```bash
python tools/verify_assets.py
python -m unittest discover -s tests -v
python encode.py --help
python decode.py --help
```

The CPU contracts cover release binding, mocked CLI failure/retry paths,
transactional no-clobber output, input size budgets, regression scheduling and
artifact pairing, trusted asset rejection, and dataset identity. They do **not**
prove CUDA compilation, TorchSparse installation, GPU numerical correctness,
performance, or security against arbitrary hostile streams. GitHub Actions saves
CPU logs and environment information as a 30-day artifact; archive release
validation artifacts separately if they must be retained longer.

## CLI input and output behavior

Directory encoding validates unique unsigned 32-bit numeric frame IDs before
GPU setup and sorts them numerically. `1.bin`, `2.bin`, `10.bin` are supported;
`1.bin` and `01.bin` together are rejected as duplicate IDs.

Encoding and decoding stage output privately next to the destination. Only a
complete output is published, without overwriting existing files/directories.
Failure removes this invocation's staging; unrelated existing `.partial` files
are not deleted. Directory publication uses Linux `renameat2(RENAME_NOREPLACE)`;
unsupported platforms/filesystems fail safely instead of risking an overwrite.
File publication requires same-filesystem hard-link support. A process killed
without cleanup can leave its private staging directory; inspect ownership before
removing it. Successful publication does not claim crash-proof durable storage.

`decode.py --max-frame-bytes` defaults to 67108864 and limits the encoded frame
**body**, excluding its 340-byte identity/length/CRC wrapper. The same body budget
applies to single frames and sequence records. This is not a total decoded
CPU/GPU-memory budget; see SECURITY.md. Sequence output becomes visible only after
its final count/checksum has validated, rather than exposing an incomplete final
directory.

## Independent GPU regression

Use the supported Linux/sm_89 environment in README. Encoding and decoding must
run in separate processes. Each encode run needs a new or empty output directory.
For example, replace the data paths with a small trusted fixture set:

```bash
python tools/gpu_regression.py --codec-root . --output results/q16 \
  --mode encode --quantization-mm 16 --default-profile --bounded-records \
  --input data/000000.bin data/000001.bin data/000002.bin --repeat 12
python tools/gpu_regression.py --codec-root . --output results/q16 \
  --mode decode --quantization-mm 16 --default-profile --bounded-records --repeat 12
```

Repeat for 8, 16, 32 and 64 mm. The revised `--repeat` means **additional checked
replays after one complete pass**, not the old loop's iteration budget and not
repeats per frame. Use at least the input-frame count for `--repeat` if every
frame should also be replayed. Reports include supplied/actually tested frames,
actual iterations and checked replays. Inspect these values rather than assuming
an input-list length proves coverage.

A completed `encode.json` records packet/reference hashes and trace identity.
Decode validates that manifest and rejects incomplete runs, altered artifacts and
unexpected stale files. Reference files follow actual packet paths, including
indices above 999. Old regression output without the new manifest must be
regenerated. Decode reruns invalidate their previous success report before
preflight. The manifest is a consistency record, not authentication of untrusted
files or a replacement for running the test.

Keep the final JSON reports, exact input manifest, source revision, GPU model,
driver, toolkit and installed-package versions together. GPU tests are manual;
this repository does not assume a configured GPU Actions runner. No new GPU
performance or compatibility result is claimed by the tooling changes.

## Portable evaluation manifests

```bash
python tools/build_test_manifest.py --sequences /path/to/sequences \
  --output evaluation-every20.json --sample-every 20 --sample-offset 0
```

The default `--sample-every 1` includes every frame. Sampling is one global slice
of lexicographically sorted relative frame paths across sequences 11–21, not a
slice restarted in each sequence. Actual counts depend on the supplied complete
dataset; the tool does not hard-code 1,018 or assert a historical run used this
exact manifest.

Schema `semantickitti-test-v3` includes frame and timestamp content SHA-256 values,
selection semantics and actual available/selected counts. `dataset_sha256`
excludes the machine-local root and is portable across copies of identical data.
It binds the full input dataset, including frames not selected by sampling.
`manifest_sha256` is the exact written JSON checksum and includes the local root;
it is expected to differ across machines. Old v2 hashes are not v3 dataset IDs.
Sequence split declarations do not prove training provenance; the manifest marks
`training_split_verified=false`. Publish the actual sampled manifest and results
when reporting an experiment, after considering any local-path privacy concerns.

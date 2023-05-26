# Performance

To measure individual timings, run

```shell
RUSTFLAGS="--cfg measure" cargo run --release
```

On `tree.jpg`:

```
Image size:         1024 × 482 = 493568 pixels

Building the graph: 56 ms
Oversegmentation:   167 ms
Segment size:       32 ms
Label extraction:   10 ms
Duration:           266 ms

Num. segments:      715
```

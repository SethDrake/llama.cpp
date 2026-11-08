# MoE cache

When the experts of MoE layers stay in host memory (`-cmoe`, `-ncmoe`, `-ot`), the MoE cache keeps the most used ones in device memory for small batches (decode, speculative decoding). Large batches still upload the experts they use.

| Argument | |
|---|---|
| `--moe-cache-mib N` | cache size in MiB, split among the GPUs like the layers (default: 0, disabled) |
| `--moe-cache-cpu-share F` | share of the cache misses computed on the CPU from the host experts instead of uploaded, once the cache is full (default: 0) |
| `--moe-cache-policy {slru,lfu}` | replacement policy (default: slru), `lfu` needs `--moe-cache-cpu-share` > 0 |

The cache hits run on the device. With `--moe-cache-cpu-share F` the CPU computes a share `F` of the misses while the device computes the hits and uploads the other misses into the cache. This is faster when the CPU reads the host experts faster than the device can upload them, for example with fast RAM, a PCIe 4 link or experts that are not in pinned memory. The host experts are pinned with `-lm none`, `-lm mlock` and `-lm dio`, unless `--no-host`, but not when they are mmap'd (the default). The device uploads pinned experts at the full speed of the link, so the CPU share often does not help then. Until the cache is full every miss is uploaded, so the cache fills as fast as without the CPU share. The best share depends on the machine, the model and the cache size.

With `--moe-cache-policy lfu` the share is not used: the cache counts how often each expert is selected, halving the counts every 10 lookups per slot so that they follow the recent routing, and once it is full a miss is uploaded only if its count beats the least used cached expert that the batch does not select by more than 2. The other misses are computed on the CPU. Rarely used experts then no longer evict frequently used ones. On a 48-layer model with 512 experts per layer and 2.4% of the experts cached, a routing trace replay raised the decode hit rate from 24.6% to 30.8% and cut the uploads by a third. The routing of such models changes within a few tokens, so a long-term popularity profile predicts it poorly: the best static set of experts hit only 12% on the same trace.

## Picking the arguments

Put the script in front of your server command:

```bash
python3 scripts/moe-cache-tune.py -- build/bin/llama-server -m model.gguf -cmoe -t 8 -c 32768
```

It starts the server 6 times, at two cache sizes × shares 0, 0.5 and 0.95, and times the generation of a few thousand tokens. That takes a few minutes, mostly model loading.

From these runs it fits the upload and CPU speeds of the machine, then replays the routing of all the runs through a copy of the cache logic for other sizes and shares. It prints the measured speeds, the predicted t/s and the recommended arguments: the lowest share within 5% of the fastest.

- **Server command:** keep your own arguments (`-t`, `-c`, expert placement, load mode); the speeds and the free device memory depend on them.
- **`--max-mib N`:** tunes a smaller cache than what fits on the device. The runs in `--out` are reused while the command stays the same, so trying another value is instant.
- **`--markdown`:** prints Markdown tables to paste in a PR.

## Profiling

Two environment variables print host timings when the server exits cleanly (with `--log-verbosity 4`):

- **`LLAMA_MOE_CACHE_TIMING=1`:** the copy callback of the cache for single-token batches: reading the selected experts and the CPU part inputs (`pre`), waiting for the device, the cache policy, the uploads and the slot map copy.
- **`GGML_SCHED_TIMING=1`:** per decode graph, the time spent copying split inputs (waits for the other backend included) and computing device and CPU splits; it also prints the split layout of one decode graph.

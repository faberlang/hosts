# Q4_K GPU exactness receipt (PB-8 three-leg proof, made durable)

Audit `3c2a4769` (P2, mail `13fcb0b6`) found the PB-8 "GPU Q4_K path is
llama.cpp-exact three ways" claim rested only on mutable `/tmp` artifacts.
This receipt freezes that evidence into committed fixtures pinned by
`macos-arm64/tests/q4_k_exactness_pins.rs`, reproducible from a clean
worktree with `cargo test -p faber-host-macos-arm64 --test q4_k_exactness_pins`.

## Source material (2026-08-23 capture)

- Model: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
  sha256 `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653`
- Tensor: `blk.2.ffn_down.weight`, GGUF dims `[4864, 896]`, GGML type 12
  (`Q4_K`), region offset 274201664, 2451456 bytes (896 rows × 19 superblocks
  × 144 B), region sha256
  `92ecbb7216a294133f4a41a34891a3b276aeac6ccfb8c924355c39f66b18b4c1`.
- GPU trace: instrumented-host dump `/tmp/pb8-trace/dump/pid48884-step0`
  (`FABER_HOST_TRACE_DIR`; the run behind `qwen-oracle-after.json`).
  Row 0 of `371-prefill_blk2_hh` is the ffn_down GEMV input; row 0 of
  `372-prefill_blk2_down` is the GPU output; `90-blk_2_ffn_down_weight` is
  the uploaded packed weight buffer.
- Reference conversion: `llama-quantize --allow-requantize <model> <out> F32`
  (homebrew llama.cpp b10150-era build), full-file sha256
  `94661e946a274076d4f15e14cbc06f68bcda354136c729aec82875a1267511af`.
- Emitted-GPU dequant semantics: `q4_k_chunk_statements`
  (`radix/crates/radix-mir-metal/src/emit/quantized_matmul.rs`), mirrored by
  the hosts CPU body `gemv_q4_k`
  (`macos-arm64/src/kernel/library.rs`) — same per-element formula
  `d·sc·nib − dmin·mn` in the same evaluation order as llama.cpp
  `dequantize_row_q4_K`.

## Fixtures (columns 62–69 of the GEMV, i.e. GGUF rows 62–69)

| File | Bytes | sha256 |
| --- | --- | --- |
| `q4k-cols62-69.gguf.bin` | 21888 | `88b3a4616b34720ea43f31746c10252867fb3aa3ba917b007be307a86ebb56fc` |
| `q4k-cols62-69.device.bin` | 21888 | `88b3a4616b34720ea43f31746c10252867fb3aa3ba917b007be307a86ebb56fc` |
| `hh-row0.f32` | 19456 | `cc038d21854de70ec1144878a7f9c221f3ec1835b218254c4ac0895a3337770f` |
| `gpu-down-row0-cols62-69.f32` | 32 | `9c0e68460da0e2c5acb4c76c7387df6475744205d121e376259878633b99bb3b` |
| `llama-f32-cols62-69.f32` | 155648 | `85d985312adeb791fbc6c7632e6c3ecafcf813baae5d0a955d8329c9ee0408d6` |

## The three legs (as pinned by the test)

1. **MSL-semantics simulation vs GPU trace.** The hosts Q4_K GEMV body
   (arithmetic mirror of the emitted MSL chunk body) over the committed
   GGUF-column bytes and the committed trace input reproduces the committed
   GPU output values within the accumulation-order band. Observed at capture:
   max abs deviation 6.1e-4 at column 62 (709.2916 vs GPU 709.2922); the
   PB-8-era numpy sim observed the same relationship (sim 939.69764 vs GPU
   939.69769 on the PB-7 trace).
2. **Independent dequant vs llama-quantize F32.** Dequantizing the committed
   packed columns element-for-element is bit-exact (f32 bit equality,
   38912/38912) against the committed llama-quantize F32 reference. Verified
   additionally at capture over the whole tensor region (0.0 max diff).
3. **Uploaded device bytes vs GGUF region.** The committed device-capture
   bytes are byte-identical to the committed GGUF-region bytes (equal
   sha256); at capture the full 2451456-byte uploaded weight buffer equaled
   the full GGUF region (both sha256 `92ecbb72…`).

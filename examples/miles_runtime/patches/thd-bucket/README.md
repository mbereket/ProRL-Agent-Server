Opt-in: `MILES_THD_PAD_BUCKET=N` rounds each packed THD micro-batch (per CP rank) up to a multiple of N tokens
(a pad sequence at the end, as Miles already does up to TP*128). Without the env var the patch is a no-op.
Purpose: kernels that JIT-compile/autotune per tensor shape (GDN chunk kernels) otherwise meet a new length every
micro-batch on real agentic data. Keep --max-tokens-per-gpu a multiple of N so padding never exceeds it.

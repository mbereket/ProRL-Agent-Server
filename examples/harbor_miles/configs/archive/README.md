# Archived configs (superseded; kept verbatim for the record)

These flat configs ran on older launcher commits, with older defaults (adapter serving, LR 1e-5, zero-variance groups kept,
logprob chunk 4096, MTP loss ON before 2026-10-01 ~06:00, no CPU reserve) and older paths (task lists in `configs/`, run dirs
under `<HM_ROOT>/runs`). On harbor-miles they would pick up the recipe defaults and break on moved paths, so do NOT run them
from here. To reproduce one exactly, check out the commit below (`git worktree add <dir> <commit>`) and submit from there.
To re-run the same EXPERIMENT on the current recipe, write a new file in `configs/experiments/` (e.g.
`experiments/9b-swegym-overfit8.env` replicates D2). Evidence these runs produced: `miles-work/FINDINGS.md`.

| dir | configs | ran on (last commit touching the file) | evidence / why archived |
|---|---|---|---|
| `swegym-9b/` | `swegym-codex-rand48-lora-1n{,-lr1e-4,-lr3e-5}.env`, `-lr3e-5-if96.env`, `-2n.env`, `swegym-codex-rand48-base-eval.env` | miles-path-b 86aa288e / 4053fff1 / 64caaea4 (09-30 23:04 - 10-01 00:39) | D1 head-to-head + LR sweep (A-parity settings: adapter serving, LR sweep 1e-5/3e-5/1e-4, eval 48x2 every 5 steps); D1 FINAL = Path B |
| `swegym-9b/` | `diag-overfit8-lr3e-5-olzero.env`, `-olverifier.env`, `diag-overfit8.txt` (= `tasks/swegym-overfit8.txt`) | miles-diag 57674a7d (rebased copy; the runs were dfw 19601498 / 19608462 / 19608790) | D2 (learning: +.23 +- .06 paired, 8/8 tasks up at steps 20-26) and the overlong-policy comparison; MTP loss ON |
| `swegym-9b/` | `swegym-codex-rand48-ref9b.env` | miles-path-b 7af76c46 | 9B reference run (recipe from step 6, NO_MTP=0) |
| `swegym-9b/` | `diag-workload9b-{96k,128k}.env` | miles-diag 57674a7d | 9B SWE-Gym workload dumps at 96k/128k (STACK's REAL trainer-replay traces) |
| `swegym-9b/` | `swegym-codex-lora-{1n,2n}.env`, `swegym-mswea-lora-1n.env`, `fleet-smoke-rand48-merged-1n.env` | miles-path-b 3d41d86e / 1b320023 / c68a1791 / 864cb7ea | early SWE-Gym bring-up (93-task pool, mini-swe-agent), FLEET readiness smokes |
| `swegym-27b/` | `swegym-codex-27b-lora-{1n,2n}.env` | miles-path-b b8d7e8c2 / 6bad3530 | 27B on SWE-Gym (D5 bring-up); superseded by de4 |
| `de4-27b/` | `de4-codex-27b-lora-{1n,2n}.env`, `-lora-colo1n.env`, `-smoke2n.env`, `-rollout128k.env` | miles-path-b 0ad57d44 / 11cf655a / 1fdfa6d6 | first de4 27B configs (64k, provisional in-flight); replaced by `layouts/` + `experiments/` |
| `de4-27b/` | `de4-codex-27b-bs128-2n.env` | miles-path-b b8a7f86a | batch point 2; re-expressed as `experiments/de4-codex-27b-bs128-2n.env` (equivalent) |
| `de4-27b/` | `diag-de4-27b-overfit8-2n-r3.env` (+ `-2n.env`, `-2n-t8.env`, `-1n.env`) | miles-diag 7f76aa95 (r3; earlier variants b3cc86b2 / e840068e) | DIAG 27B de4 overfit r3 (dfw 19612545); re-expressed as `experiments/diag-de4-27b-overfit8-r3.env` (equivalent) |
| `de4-27b/` | `q27-de4-overfit8-2n-lr1e4.env` (+ `-split1n-lr1e4.env`, `-colo1n.env`, `q27-de4-colo1n-smoke96k{,-r2}.env`, `de4-codex-27b-base128k.env`) | miles-qwen27b d48fac8a / 335b1355 / b1746c33 / 6331c7f4 (LOCAL branch in miles-work/qwen27b/ProRL-pb, not pushed) | LR-1e-4 hedge (dfw 19612902; re-expressed, equivalent), 1n split / colocated timing arms (hel 1527509, aws-iad 7599199), colocated smoke (aws-iad 7598908), the 27B de4 base pass (hel 1527107/1527112) |
| `bbh/` | `learn-bbh8-async-{1n,2n}.env`, `smoke-bbh8-1n.env`, `bbh-overfit8.txt` | miles-path-b 5b099042 | bbh/bbh-mcp is no longer used (user, 2026-09-30) |

`de4-27b/EQUIVALENCE.md`: the dry-render proof that the three re-expressed running configs launch the same run.

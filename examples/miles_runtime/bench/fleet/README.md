# FLEET tools (A100 feasibility, 2026-09-30). Submit with jobs/submit.py --pkg <this dir> (exported as $SCOMPOSE_PKGS/fleet).
# probe.sh       host facts + Slurm QOS + runtime setup + Qwen3.5-9B staging on a new cluster
# serve9b.sh     SGLang Qwen3.5-9B TP1/TP2/TP4 side by side x agentic multi-turn sim (SWE-Gym shape)
# te_attn_test.py / te_probe.sh  which Transformer Engine attention backend works for Qwen3.5 attention on this GPU
# sim_multiturn.py  copied from the qwen27b agent (agentic multi-turn rollout simulator)

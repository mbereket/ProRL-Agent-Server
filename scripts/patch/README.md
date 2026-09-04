# scripts/patch — superseded

These scripts applied local changes to slime v0.3.0 and SGLang 0.5.13 at setup
time. The same changes now live as commits on the polar forks that
`examples/harbor_slime_grpo/internal/setup/stack/uv.lock` pins:

| script | fork commit |
|---|---|
| `patch_slime_router_tokens.sh` | mbereket/slime `polar-v0.3.0`: "agent adapter: return exact SGLang token metadata in chat-completions responses" |
| `patch_slime_engine_base_port.sh` | mbereket/slime `polar-v0.3.0`: "rollout: SLIME_ENGINE_BASE_PORT overrides the SGLang engine base port" |
| `patch_sglang_0513_token_metadata.sh` | mbereket/sglang `polar-v0.5.13`: "openai chat completions: emit token ids and meta_info when logprobs are requested" |

They are kept as a readable record of the delta and for stacks that install
upstream slime/sglang from PyPI; the harbor example no longer runs them.

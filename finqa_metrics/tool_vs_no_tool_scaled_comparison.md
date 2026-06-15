| Method | Answer Source | JSON/Tool | Program Exec | Exact | Num@1 | Num@5 | Scaled Num@1 | Scaled Num@5 | Scaled Exec@5 |
|---|---|---|---|---|---|---|---|---|---|
| No tool: Base Qwen2.5-7B | direct answer | 98.41% | 0.00% | 18.80% | 45.54% | 57.67% | 48.74% | 61.56% | - |
| No tool: SFT-full | direct answer | 100.00% | 97.06% | 39.52% | 58.47% | 70.37% | 62.47% | 75.17% | 74.56% |
| No tool: Clean balanced DPO | direct answer | 99.89% | 96.71% | 39.75% | 58.92% | 70.71% | 62.81% | 75.40% | 75.26% |
| No tool: GRPO r4 hard exec | direct answer | 100.00% | 96.72% | 40.09% | 58.58% | 70.59% | 62.47% | 75.29% | 74.94% |
| No tool: GRPO r5 exec consistency | direct answer | 100.00% | 96.94% | 39.30% | 58.12% | 70.14% | 61.90% | 74.71% | 74.65% |
| Tool-call: SFT rank16 | calculator | 100.00% | 96.94% | 19.59% | 27.52% | 28.65% | 65.91% | 74.29% | 74.29% |
| Tool-call: SFT rank32 | calculator | 100.00% | 96.49% | 19.03% | 27.63% | 28.65% | 65.12% | 73.27% | 73.27% |
| Tool-call: GRPO r32 ckpt1200 | calculator | 100.00% | 98.41% | 20.05% | 29.78% | 30.92% | 67.38% | 75.65% | 75.65% |
| Tool-call: GRPO r32 ckpt2400 | calculator | 100.00% | 98.53% | 20.72% | 30.58% | 31.82% | 67.72% | 76.10% | 76.10% |
| Tool-call: GRPO r32 final6000 | calculator | 100.00% | 98.64% | 20.84% | 30.69% | 31.82% | 68.18% | 76.44% | 76.44% |

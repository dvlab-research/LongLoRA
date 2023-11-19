# Evaluation on LongBench and L-Eval Benchmarks

We evaluate our supervised fine-tuned model, [LongAlpaca-7B-16k](https://huggingface.co/Yukang/LongAlpaca-7B-16k), on LongBench and L-Eval benchmarks.

Table - Evaluation on LongBench English tasks
| Model | Avg | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot Learning | Code | Synthetic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-3.5-Turbo | 44.0 | 39.8 | 38.7 | 26.5 | 67.1 | 54.1 | 37.8 |
| Llama2-7B-chat | 31.0 | 24.9 | 22.6 | 24.7 | 60.0 | 48.1 | 5.9 |
| Ours | 36.8 | 28.7 | 28.1 | 27.8 | 63.7 | 56.0 | 16.7 |

The predictions can be found [here](https://github.com/dvlab-research/LongLoRA/tree/main/benchmarks/Pred_LongBench).


Table 2 - Evaluation on L-Eval open-ended tasks, comparing to GPT-3.5-Turbo and judging win rates via GPT-4.
| Model | Win-rate | Wins | Ties |
| --- | --- | --- | --- |
| Ours | 39.06 | 45 | 60 |

The predictions can be found [here](https://github.com/dvlab-research/LongLoRA/tree/main/benchmarks/Pred_L-Eval).

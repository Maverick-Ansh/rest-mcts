# ReST-MCTS\*, built from scratch on 2× T4

A from-scratch rebuild of **ReST-MCTS\*: LLM Self-Training via Process Reward Guided Tree Search**
(Zhang et al., NeurIPS 2024, [arXiv 2406.03816](https://arxiv.org/abs/2406.03816),
official code [THUDM/ReST-MCTS](https://github.com/THUDM/ReST-MCTS)), run in one Colab notebook on a
Kaggle 2× Tesla T4 backend.

Policy: `Qwen2.5-1.5B-Instruct` (vLLM). Value model: the same backbone with LoRA and the official
head, `Linear(vocab, 1)` on the last-token logits. Task: MATH.

**Status: in progress.** Parts 1 to 5 are built. The MCTS\* budget runs and the self-training loop (Part 6) are not finished.

## What the notebook builds

| Part | Component | Paper |
|---|---|---|
| 1 | Quality value `v_k` and weighted reward `w_k`, plus value inference on a verified tree | Eq. 1–2, Thm 1, §3.2, Fig. 3 |
| 2 | Step-level policy on a vLLM server with prefix caching | `get_next_step` |
| 3 | Value data D_V0 from BFS trees, then the value model and its training | App. B.1, E.1, Eq. 37–38 |
| 4 | MCTS\*: UCB selection, self-critic, expansion, greedy MC rollout, weighted backprop | App. C.1, Alg. 2 |
| 5 | Search at a matched token budget vs Self-Consistency / Best-of-N | Fig. 2, Table 3 |
| 6 | Mutual self-training vs ReST-EM | Alg. 1, Table 2 |

## Results so far

* **Fig. 3 reproduced exactly.** The paper's dice tree gets the same `m`, `w` and `v` on all 11 nodes. Theorem 1 holds on 1M random cases.
* **Step format matters at 1.5B.** Greedy MATH-500 accuracy is 59.0% with Qwen's free-form prompt, but only 51.5% or 44.0% when the model is forced into "Step k:" lines. So a step here is one paragraph.
* **π0 greedy on MATH-500 (all 500): 55.0%.** By level, L1 93% to L5 22%.
* **D_V0 with zero human labels:** 400 BFS trees and 18,155 (partial solution, v) targets.
* **V0 on held-out questions:** ±0.1 accuracy (Eq. 38) is 37.7%. For comparison, a constant predictor gets 14.9% and depth alone gets 23.0%; the paper reports 69.3% on its 473k-sample set. V0 ranks correct final answers above wrong ones with **AUROC 0.883**. Calibration is squashed: a truly correct solution scores about 0.78.
* **The self-critic fails at 1.5B.** On prefixes known to be unfinished, the official "solved/unsolved" critic says "solved" 32–70% of the time, and the App. F advice critic does so 48–68% of the time. So MCTS\* ends a branch only on a `\boxed{}` answer or EOS.
* **Baselines on 200 MATH-500 problems, N = 1 → 16 samples:** Self-Consistency 51.0 → 64.5%. Value-weighted voting reaches 65.0%. Best-of-N by last-step value reaches 58.5%. The paper's Best-of-N by the product of step values drops 51.0 → 44.5%: because `v` is a progress value, the product rewards short solutions.
* **Engineering:** served through HF, the value model was the bottleneck (6.4 nodes/s). Serving it through vLLM fixed that. Merge the LoRA, collapse the vocab head to the 1536-d hidden-state probe `W_U^T w`, and turn on prefix caching. It matches the HF model within 0.0012.

## Still to do

* MCTS\* at T = 20 and T = 50 on the same 200 problems (the Fig. 2 comparison).
* Self-training iteration 1 and then 2: MCTS\* data, SFT π_M and π_EM, then V1 and V2, then greedy evaluation on MATH-500.

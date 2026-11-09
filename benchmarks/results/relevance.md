# Relevance evaluation

Free-text queries (syntax characters escaped), ranked OR retrieval, top 1000. MAP is over the top 1000. nDCG@10 uses graded judgments where available (Cranfield: gain = 5 - code).

## Cranfield (1,400 docs, 225 judged queries)

| Model | P@10 | R@10 | MAP | nDCG@10 | MRR |
|---|---:|---:|---:|---:|---:|
| TF-IDF (sqrt tf) | 0.2356 | 0.4005 | 0.3078 | 0.3551 | 0.5294 |
| TF-IDF (log tf) | 0.2391 | **0.4095** | 0.3083 | 0.3572 | 0.5263 |
| BM25 (k1=1.2, b=0.75) | 0.2338 | 0.3966 | 0.3063 | 0.3559 | 0.5316 |
| BM25 tuned on test (k1=2.0 b=0.75) † | **0.2413** | 0.4035 | 0.3106 | **0.3653** | 0.5327 |
| BM25, no stemming | 0.2240 | 0.3820 | 0.2822 | 0.3373 | 0.5164 |
| BM25, no stopword removal | 0.2311 | 0.3924 | 0.3056 | 0.3577 | 0.5405 |
| BM25, larger stopword list | 0.2387 | 0.4017 | **0.3158** | 0.3611 | **0.5420** |

† Parameters chosen on the evaluation queries themselves: an optimistic upper bound, not a fair comparison.

Paired randomisation test (20,000 sign flips, two-sided):

| Comparison | Δ (BM25 − other) | p-value |
|---|---:|---:|
| BM25 (k1=1.2, b=0.75) vs TF-IDF (sqrt tf) [nDCG@10] | +0.0008 | 0.9121 |
| BM25 (k1=1.2, b=0.75) vs TF-IDF (sqrt tf) [AP] | -0.0015 | 0.8203 |
| BM25 (k1=1.2, b=0.75) vs TF-IDF (log tf) [nDCG@10] | -0.0013 | 0.8738 |
| BM25 (k1=1.2, b=0.75) vs TF-IDF (log tf) [AP] | -0.0020 | 0.7756 |
| BM25 (k1=1.2, b=0.75) vs BM25, no stemming [nDCG@10] | +0.0186 | 0.0378 ✱ |
| BM25 (k1=1.2, b=0.75) vs BM25, no stemming [AP] | +0.0241 | 0.0004 ✱ |

✱ significant at α = 0.05

BM25 nDCG@10 over k1 × b:

| k1 \ b | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---:|---:|---:|---:|---:|
| 0.4 | 0.3034 | 0.3105 | 0.3237 | 0.3283 | 0.3275 |
| 0.8 | 0.3138 | 0.3313 | 0.3445 | 0.3494 | 0.3473 |
| 1.2 | 0.3199 | 0.3413 | 0.3535 | 0.3559 | 0.3519 |
| 1.6 | 0.3213 | 0.3490 | 0.3600 | 0.3615 | 0.3543 |
| 2 | 0.3255 | 0.3517 | 0.3631 | **0.3653** | 0.3548 |

## SciFact (5,183 docs, 300 judged queries)

| Model | P@10 | R@10 | MAP | nDCG@10 | MRR |
|---|---:|---:|---:|---:|---:|
| TF-IDF (sqrt tf) | **0.0927** | 0.8383 | 0.6231 | 0.6752 | 0.6356 |
| TF-IDF (log tf) | **0.0927** | **0.8400** | 0.6087 | 0.6647 | 0.6211 |
| BM25 (k1=1.2, b=0.75) | 0.0903 | 0.8168 | 0.6407 | 0.6828 | 0.6500 |
| BM25 tuned on test (k1=2.0 b=0.75) † | 0.0917 | 0.8277 | **0.6469** | **0.6909** | **0.6590** |
| BM25, no stemming | 0.0870 | 0.7823 | 0.6293 | 0.6664 | 0.6407 |
| BM25, no stopword removal | 0.0893 | 0.8138 | 0.6429 | 0.6834 | 0.6527 |
| BM25, larger stopword list | 0.0910 | 0.8235 | 0.6430 | 0.6863 | 0.6526 |

† Parameters chosen on the evaluation queries themselves: an optimistic upper bound, not a fair comparison.

Paired randomisation test (20,000 sign flips, two-sided):

| Comparison | Δ (BM25 − other) | p-value |
|---|---:|---:|
| BM25 (k1=1.2, b=0.75) vs TF-IDF (sqrt tf) [nDCG@10] | +0.0076 | 0.4159 |
| BM25 (k1=1.2, b=0.75) vs TF-IDF (sqrt tf) [AP] | +0.0176 | 0.0827 |
| BM25 (k1=1.2, b=0.75) vs TF-IDF (log tf) [nDCG@10] | +0.0181 | 0.0665 |
| BM25 (k1=1.2, b=0.75) vs TF-IDF (log tf) [AP] | +0.0320 | 0.0033 ✱ |
| BM25 (k1=1.2, b=0.75) vs BM25, no stemming [nDCG@10] | +0.0164 | 0.0765 |
| BM25 (k1=1.2, b=0.75) vs BM25, no stemming [AP] | +0.0114 | 0.2411 |

✱ significant at α = 0.05

BM25 nDCG@10 over k1 × b:

| k1 \ b | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---:|---:|---:|---:|---:|
| 0.4 | 0.6642 | 0.6685 | 0.6723 | 0.6740 | 0.6756 |
| 0.8 | 0.6649 | 0.6764 | 0.6802 | 0.6800 | 0.6824 |
| 1.2 | 0.6674 | 0.6763 | 0.6790 | 0.6828 | 0.6805 |
| 1.6 | 0.6692 | 0.6820 | 0.6850 | 0.6876 | 0.6852 |
| 2 | 0.6614 | 0.6780 | 0.6826 | **0.6909** | 0.6881 |

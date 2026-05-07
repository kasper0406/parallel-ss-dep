| Run | Final val CE | Final val PPL | Final val KL | Wallclock |
| --- | ---: | ---: | ---: | ---: |
| KL+CE (`alpha = 0.5`, top-K = 20) | 1.9333 | 6.91 | 1.0017 | 170s |
| CE-only baseline | 1.7921 | 6.00 | 0.0000 | 168s |

**KL+CE vs CE-only: 15.2 % PPL** — kl+ce worse than ce-only.

**Verdict: NEEDS CORPUS FIX.** KL+CE is materially worse than CE-only by >10% PPL. Likely teacher-data misalignment (Phase 15 redux). Do NOT scale up; investigate corpus / loss weights / top-K coverage.

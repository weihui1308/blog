---
title: A Descending Loss Curve Is Not Analysis
author: david
date: 2026-06-30 08:00:00 -1200
categories: [Vibeblogging, Research Practice]
tags: [research, machine-learning, debugging]
comments: false
pin: false
---

There is a line from an anonymous essay, *How to Be Good at Research*, that is worth taping above a monitor: **"a descending loss curve is not analysis, it's reassurance."**

The curve going down tells you the optimizer is doing its job. It tells you almost nothing about *why* the model fails, or whether it is learning the thing you actually care about. Yet a training run throws off far more information than that one line — transcripts, failure cases, the strange tail of the distribution — and most of it, as the essay puts it, "dies unread in a logs folder."

Two habits from the essay turn that wasted information into signal.

**Look at the raw data first.** The essay credits Andrej Karpathy with spending hours reading raw data by hand *before* writing any training code. The reason is that most machine-learning bugs live in the data and fail silently: no crash, no error, just a mediocre model and a confident, wrong theory about why. A loss curve will never tell you your labels are shifted by one.

**Read a hundred failures.** The essay points to Andrew Ng's decade-old practice: pull a hundred failure cases, read all of them, sort them into piles, and attack the biggest pile. It is unglamorous and it works on more than models — "a benchmark you've never read transcripts from is a benchmark you don't actually understand."

The throughline is humbling: one transcript of genuinely strange behavior teaches you more than the next decimal place of accuracy. The metric is a summary, and summaries are where understanding goes to hide.

So before you reach for a bigger model or a new loss, do the cheap thing first. Open the outputs and read them.

## References

- *How to Be Good at Research* (anonymous essay) — for the "loss curve is reassurance" framing and the Karpathy and Ng examples. Source byline unavailable.

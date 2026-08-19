# Experiments

Things that were tried and measured, so nobody spends an afternoon
rediscovering them. Each one still has a runnable script in `scripts/`; the
numbers below are the summary.

## Prefix the question before compressing

`scripts/experiment_question_aware.py` -- **no reliable gain, not adopted.**

The encoder is bidirectional, so a question placed in front of a window is
visible to every document token in it, and question-relevant words ought to
score higher.

    ratio  n     agnostic EM/F1   question-aware EM/F1   delta
    0.335  300   0.570 / 0.701    0.573 / 0.707          +0.003 / +0.006
    0.211  300   0.433 / 0.539    0.457 / 0.559          +0.024 / +0.020
    0.209  750   0.429 / 0.547    0.441 / 0.548          +0.012 / +0.001

The middle row reads as a real gain at tight budgets, and it is the row to
distrust: at 750 questions the exact-match gain halves and the F1 gain
disappears. The standard error there is about 0.018, so +0.012 sits inside it.

The mechanism explains the null. The model never saw a question during
training, so nothing taught it to raise keep-probabilities for question-relevant
words -- attention alone supplies no objective. Conditioning on the question is
a training change, not an inference one.

It is not free either: one compression per question rather than per document,
so a compressed document can no longer be cached across queries.

## Keep whole sentences instead of scattered words

`scripts/experiment_sentence_level.py` -- **clearly worse, not adopted.**

At ratio 0.21, word-level top-k leaves a fifth of the words with none of the
syntax that connected them. Keeping fewer, complete sentences should read
better on the same budget.

    method          EM      F1
    word-level      0.429   0.547
    sentence-level  0.320   0.406

Not close. A fifth of the words spent on whole sentences buys only a fifth of
the sentences, so most facts are simply absent; word-level keeps a trace of
every sentence. For extractive QA, coverage beats coherence.

## Run rules first, then the classifier

`scripts/experiment_cascade.py` -- **holds up, not yet promoted.**

`rules` reaches 0.62 deterministically and never touches content. Reaching 0.21
from there asks the classifier to keep a third of a cleaner input rather than a
fifth of a raw one. Measured on MeetingBank QA at ~0.22, answers judged by
Sonnet 4.6:

    n(questions)  classifier only   rules -> classifier   delta
    450           0.484 / 0.603     0.500 / 0.617         +0.016 / +0.014
    900           0.477 / 0.595     0.508 / 0.612         +0.031 / +0.017

Better on both metrics at a slightly tighter ratio, and the gap widens with
sample size rather than shrinking -- which is the test the question-prefix
experiment failed.

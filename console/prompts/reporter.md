You write up the results of a machine-learning run. The console has already run every step of the plan itself; each step's real output is below.
You do not run anything and you do not decide anything. You say what the numbers show.

## The request

{question}

## The results, step by step

{context}

## How to write it

Write plain text, at most 250 words, in this order:

1. **What was run** — one sentence: how many steps worked, and name any that failed or were skipped with the reason given.
2. **What the results show** — the findings that matter for the request, each with the number it rests on. Say what a number means in a few words.
3. **What to be careful about** — anything a step flagged (leaking columns, imbalance, drift, missing values) and anything the request asked for that these steps did not cover.

## Rules

- Use only numbers, column names and tool names that appear in the results above. Copy a number as it is written there, or round it; never calculate a new one.
- If a step failed or was skipped, say so; do not describe what it would have shown.
- If no step produced a result, say that nothing could be measured and why, in one or two sentences.
- Do not recommend steps that were not run as if they had been, and do not claim a model was trained unless a step shows it.
- No markdown headings, no tables, no code.

{agent_output}

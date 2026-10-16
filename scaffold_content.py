"""The hand-written prose behind each topic's prompts: what the problem is, when each tool is the right
one, how to work, how to judge the result. `scaffold.py` combines it with the tool catalogue read off the
code. Every tool named in a `choose` row is checked to exist, so this cannot point at a tool that was renamed.

Each topic lives in `scaffold_parts/<category>.py` as a `TOPICS` dict entry:

    slug       prefix of the topic's two agents: `<slug>_decider_agent`, `<slug>_agent`
    title      what the problem is called in a sentence
    role       what a data scientist working on it is responsible for
    fits       signs that the problem really belongs to this topic
    not_this   neighbouring topics it is easily confused with, and what tells them apart
    choose     (situation, [tool function names]) — the decision guide
    workflow   the steps the worker follows
    judging    how to tell a good result from a flattering one
    produce    what the final answer must contain
    rules      topic-specific rules
    decide     what the decider checks before choosing
    decide_rules  topic-specific rules for the decider
"""

from scaffold_parts.semisupervised import TOPICS as SEMISUPERVISED
from scaffold_parts.supervised import TOPICS as SUPERVISED
from scaffold_parts.unsupervised import TOPICS as UNSUPERVISED

TOPICS = {**SUPERVISED, **SEMISUPERVISED, **UNSUPERVISED}

---
name: video-to-skill
description: >
  Turns a tutorial or how-to video into a working Claude Code skill by cross-checking two independent
  "reads" of the video against each other before building anything — one from the `bradautomates/claude-video`
  plugin (frames + transcript), one from Gemini (native video understanding) — so a single mis-transcribed
  step or hallucinated detail can't silently make it into the generated skill. Use this whenever the user
  pastes a YouTube link, Instagram/TikTok reel, or other tutorial video and asks to "build this skill",
  "turn this into an agent", "make a skill out of this video", or otherwise wants Claude to replicate or
  automate what the video demonstrates. Do not attempt to read a video's content or build a skill from a
  video link using only your own eyes/ears — always run the full two-source cross-check in this skill first,
  even if the request looks small.
---

# Video → Skill

Build a Claude Code skill from a tutorial video by getting two *independent* reads of it, reconciling
them, and only asking the user about the parts the two reads disagree on. The reason for two sources:
a single pass over a video (yours or anyone else's) can mishear a command, miss a step off-screen, or
invent a plausible-sounding detail. Two independent reads that actually agree on a claim are strong
evidence it's real; where they disagree, that's exactly where a human's ten seconds of judgment is worth
more than either model guessing.

## Step 0 — Check prerequisites before touching the video

Do this before anything else, and don't skip it because it looks like extra ceremony — proceeding
without both sources defeats the entire purpose of this skill.

1. **`bradautomates/claude-video` plugin.** Check whether it's installed/enabled (e.g. `SearchPlugins`,
   or whatever plugin-listing tool this environment exposes). If it's missing, tell the user directly:
   they need to install it (plugin marketplace add) before this skill can produce a frames+transcript
   read. Do not substitute your own guess at the video's content — you can't reliably see video frames
   or hear audio without it.
2. **A Gemini connector.** Check `ListConnectors` / `SearchMcpRegistry` (or ask the user directly if no
   such tool exists in this environment) for a way to reach Gemini with the video — an MCP connector,
   a Slack `@googlegemini` mention, or direct API access, depending on where this session is running.
   If nothing reachable exists, say so plainly.

**If either prerequisite is missing, stop here.** Explain exactly what's missing and how to add it
(name the specific plugin/connector), and do not proceed to a single-source build — that's a materially
different, weaker deliverable than what this skill promises, and the user should get to choose it
explicitly rather than have it happen by default.

## Step 1 — Read A: claude-video (frames + transcript)

Using the claude-video plugin, pull the video's frames and transcript. Produce a numbered list of
every discrete, checkable claim the video makes — setup steps, exact commands, config values, file
paths, the order operations happen in, and any stated rationale ("we do X because Y"). Number them so
they can be referenced later (A1, A2, A3...). Be literal: write what's shown/said, not your
interpretation of intent.

## Step 2 — Read B: Gemini (native video understanding)

Send the *same* video URL to Gemini and ask it to independently produce the same kind of numbered,
literal claim list (B1, B2, B3...). Do this without showing Gemini Read A first — the value of a second
read comes entirely from its independence. If Gemini can only be reached through a chat surface (e.g. a
Slack mention) rather than a tool call, ask the user to relay the prompt and response, or do it yourself
if you have that access.

## Step 3 — Reconcile into one labeled spec

Align A and B claim-by-claim (many will restate the same step in different words — match on meaning, not
wording) and produce one merged, numbered spec where every line carries exactly one label:

- **`confirmed`** — both sources describe this the same way.
- **`single-source`** — only one source reported it. Note which one (A or B). This is not necessarily
  wrong — some steps are only visible in frames, or only audible in speech — but it's unverified.
- **`conflict`** — the two sources disagree on the same step (different command, different order,
  different value). State both versions in full.

Write this out as a numbered list, e.g.:

```
1. [confirmed] Runs `npm install claude-video` before anything else.
2. [single-source: A] Terminal is Warp, not a standard shell (only visible in frame, not stated aloud).
3. [conflict] Step order: A says the API key is set via `.env` before install; B says it's set via
   `export` after install and before running the script.
```

Save this reconciled spec to a file (e.g. `video-spec.md`) in the current working directory so there's
a record of what was confirmed vs. resolved before generation — this is the paper trail if the
generated skill later needs debugging.

## Step 4 — Surface only the conflicts

Show the user the `conflict` lines only — not the full spec, they don't need to re-review what both
sources already agree on. For each conflict, present both versions clearly and ask which is correct
(or let them supply a third answer if both are wrong). Use `AskUserQuestion` if available, or a plain
question in conversation otherwise.

Mention `single-source` lines only if there are a small number and they're consequential (e.g. an API
key handling step) — otherwise let them ride into the spec as-is; flagging every single-source line
defeats the point of narrowing review to what actually needs a human.

## Step 5 — Fold in resolutions, then hand off to skill-creator

Update the reconciled spec with the user's answers so every `conflict` line is resolved. Then invoke
the `skill-creator` skill, passing the finalized spec as the description of what the new skill should
do — skill-creator handles the actual SKILL.md authoring, frontmatter, triggering description, and
(if warranted) test cases. Don't hand-roll the generated skill's structure yourself; that's
skill-creator's job and it knows the conventions (progressive disclosure, bundled scripts/references,
description-triggering patterns) better than a one-off would.

## Notes

- This skill is explicitly about *fidelity*, not speed — the two-source cross-check exists because a
  generated skill that confidently does the wrong thing (a misheard flag, a hallucinated step) is worse
  than one that takes an extra minute to build correctly.
- If the video is short/simple enough that both reads are trivially identical, that's fine — the
  reconciliation step still confirms it rather than assuming it.
- If the user explicitly says they don't want the full cross-check (e.g. "just use the transcript, I
  don't care about double-checking"), that's their call to make — but only after you've told them the
  tradeoff, not by silently skipping Step 0/2 on your own judgment.

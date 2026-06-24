# TakeMeter — Planning

## Community

**Chosen community:** r/television and r/WidowsBay — specifically Widow's Bay (Apple TV+) Season 1 finale discussion threads

**Why this community:**
Widow's Bay is a horror-comedy limited series whose audience skews toward engaged, writerly viewers who discuss craft explicitly. The season 1 finale (Episode 10) aired in mid-June 2026, generating simultaneous reaction posts, detailed structural analysis of the show's comedy/horror balance, and a dense cluster of Season 2 prediction posts driven by the renewal announcement and the 8-bell cliffhanger. All three label types appear naturally and in roughly equal volume in the same thread window, which meant no hunting for underrepresented labels during collection. The show's mythology-heavy finale also produces prediction posts that include analytical reasoning — an ideal stress test for the analysis/prediction boundary.

**Collection window note:** Posts collected from the r/WidowsBay Episode 10 finale discussion thread and r/television Widow's Bay threads, June 2026 (within two weeks of the finale airing).

---

## Label Taxonomy

### Label 1: `analysis`
**Definition:** A post that evaluates a show's writing, direction, performances, or themes using specific, verifiable references to the show's text — scenes, dialogue, characters, structure — rather than just delivering a verdict.

**Example 1:**
> "The writing this season has been deliberately fragmented — every episode cuts away right before an emotional payoff, which mirrors the protagonist's avoidance of grief. It's frustrating on purpose. The season 2 premiere does the same thing with the funeral scene: we see the setup, then cut to three days later. This isn't lazy editing, it's structural."

**Example 2:**
> "People keep saying [showrunner] lost the plot but if you look at the episode order, every character's arc in S3 mirrors their S1 arc in reverse — the pilot ends with the group together, S3E1 ends with them separated, and so on. The symmetry is intentional and it's been there since episode one."

### Label 2: `reaction`
**Definition:** An immediate emotional or evaluative response to a specific episode or moment, with little to no supporting reasoning — the post is expressing a feeling or verdict rather than making an argument.

**Example 1:**
> "That finale absolutely destroyed me. I was not ready. This show is genuinely one of the best things on television right now."

**Example 2:**
> "OK I finally caught up on the last three episodes and I have no words. The writing, the acting, everything. Episode 8 especially. Just incredible television."

### Label 3: `prediction`
**Definition:** A speculative claim about what will happen in a future episode or season, stated as a forecast — the primary content is the prediction itself, not evaluation of something that already aired.

**Example 1:**
> "Calling it now: [character] is the mole. The show has been dropping hints since episode 3 — the phone call in the background, the look in the finale. Next season is going to reframe everything."

**Example 2:**
> "My prediction for the finale: they're going to kill off [character] to set up the spinoff. The showrunner said in an interview they wanted a 'clean break' from the original timeline. This is what that looks like."

---

## Hard Edge Cases

**Hardest anticipated edge case:**
A post like "I have a theory that Evan is immune to being sacrificed. It wouldn't make sense for the entity to kill Warren's relative. I think if Tom tells Evan it gives him the choice." straddles `prediction` and `analysis`: it includes explicit reasoning about the show's internal logic but the primary content is a forecast about a future story development.

**Decision rule:**
If the primary content of the post is a forecast about future events, label it `prediction` even if it includes supporting analysis — the analysis is in service of the prediction, not standalone. If the post is evaluating something that already aired and only gestures at future implications in passing, label it `analysis`. The key test: could you remove the speculative sentence and still have a complete evaluative post? If yes → `analysis`. If the speculation is the point → `prediction`.

**Other edge cases encountered during annotation:**
- Performance-praise posts that read like analysis ("Matthew Rhys wins my Emmy... That speech where he teased his wife. Brutal.") — no structural argument, pure emotional declaration → `reaction`
- Character-arc observation posts ("Oh I LOVE that Dale of all people is going to end the season as the person keeping most of the island's secrets") — noticing a character arc is analytical but if it's expressed as in-the-moment delight with no evaluative reasoning → `reaction`
- Theory posts that reason backward about unexplained past events ("I wonder if the reason no one knows about the council anymore is because they were tricked...") — theorizing about how a past event explains a present mystery, framed as a forward-looking explanation → `prediction`
- Short reaction posts that happen to name a specific element ("BECHIR!!" or "The sheriff...oh my God...") — naming a moment is not analysis → `reaction`

---

## Data Collection Plan

**Source(s):**
- Top posts from r/television sorted by top/month to capture evaluative analysis posts
- Active episode discussion threads for 2–3 currently airing prestige dramas (e.g. The Bear, Severance, The Last of Us) for reaction and prediction posts
- Prediction threads posted before season finales — these are explicitly tagged and cluster predictions naturally

**Target per label:**
- `analysis`: ~70 examples
- `reaction`: ~80 examples (naturally more abundant)
- `prediction`: ~60 examples

Total: ~210 examples (buffer for rejects)

**Plan if a label is underrepresented after 200 examples:**
If `prediction` is under 20% (40 examples), I'll pull specifically from pre-finale/pre-premiere threads — prediction posts cluster there. If `analysis` is under 20%, I'll add posts from r/TrueFilm and r/television's "Best Of" threads, which skew toward more structured takes.

---

## Evaluation Metrics & Why

**Metrics used:** overall accuracy, per-class precision/recall/F1, confusion matrix

**Why these and not just accuracy:**
Accuracy alone would mask a model that learned to predict the majority class (`reaction`) most of the time. Per-class F1 tells me whether the model has actually learned the analysis/prediction boundary, which is the hardest one. The confusion matrix tells me which specific label pair is failing and in which direction — a model that confuses analysis→reaction is making a different kind of error than one that confuses prediction→analysis, and fixing them requires different interventions.

---

## Definition of Success

**Concrete threshold:**
Fine-tuned model beats the Groq zero-shot baseline by at least 10 percentage points of overall accuracy, and no single class F1 falls below 0.60. If the fine-tuned model fails to meet these bars, I'll treat it as a signal that 200 examples is insufficient for these boundaries, or that the annotation is too inconsistent to train from — not a deployment-ready classifier.

---

## AI Tool Plan

**Label stress-testing:**
Give Claude the three label definitions plus the edge case rule, and ask it to generate 8–10 posts that deliberately sit at label boundaries. If any of the generated posts can't be cleanly labeled using the definitions, tighten the definitions before annotating 200 real examples. This step happens before annotation, not during.

**Annotation assistance:**
Use Claude to pre-label batches of 30–50 examples at a time by providing the label definitions and raw post text. Every pre-labeled example will be reviewed and corrected manually before committing it to the dataset. Pre-labeled examples will be tracked with a `pre_labeled` column in the CSV (value: `yes`/`no`) for disclosure. Skimming without genuine review is not the workflow — the pre-labels are a starting point, not the ground truth.

**Failure analysis:**
After fine-tuning, paste the full list of wrong predictions (text + true label + predicted label) into Claude and ask it to surface common patterns: similar post length, sarcasm, a specific label pair that keeps recurring, posts where topic signals one label but structure signals another. Before including any pattern in the README, verify it by re-reading the actual examples in that group. The LLM's pattern claim is a hypothesis, not the analysis.

---

## Difficult Annotation Examples

1. **Post:** "I have a theory that Evan is immune to being sacrificed. It wouldn't make sense for the entity to kill Warren's relative. I think if Tom tells Evan it gives him the choice." — **Considered:** `analysis` vs `prediction` — **Decision:** `prediction`. The reasoning ("it wouldn't make sense") exists to support a forecast about a future story beat. Removing the speculation ("I think if Tom tells Evan...") would leave nothing worth keeping; the reasoning is entirely in service of the forecast.

2. **Post:** "I wonder if the reason no one knows about the council anymore is because they were tricked into becoming the previous sacrifice. Like Howard the Coward figured out a way to tie the people controlling the lever to the track themselves." — **Considered:** `analysis` vs `prediction` — **Decision:** `prediction`. The post theorizes about a past unexplained event as a forward-looking explanation for a current mystery. It's speculating about a mechanism that hasn't been shown, not evaluating something the show actually depicted.

3. **Post:** "I loved the last bit though where you see Evan learning his lesson: he sticks his head into the room visibly considers going farther in and then is like nah actually the wiser choice is to get the fuck out of here and turns right around." — **Considered:** `reaction` vs `analysis` — **Decision:** `analysis`. The post describes a specific observed scene and explicitly names what it means (character learning a lesson, choosing the wiser option). It names behavior and interprets its significance, which is evaluative content, even though the writing is casual.

4. **Post:** "Matthew Rhys wins my Emmy. No matter who else wins. That speech where he teased his wife. Brutal." — **Considered:** `reaction` vs `analysis` — **Decision:** `reaction`. Praising a performance and calling a scene "brutal" is an emotional verdict. There's no argument about why the performance works or what craft choices produced the effect. The reference to a specific scene doesn't cross the analysis threshold without interpretive reasoning.

5. **Post:** "Oh I LOVE that Dale of all people is going to end the season as the person keeping most of the island's secrets" — **Considered:** `reaction` vs `analysis` — **Decision:** `reaction`. Noticing a character's unexpected narrative position is analytical in principle, but this post is expressing delight at an observation, not making an evaluative argument. No reasoning about why this is a good or interesting writing choice → `reaction`.

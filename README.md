# TakeMeter

A fine-tuned text classifier that evaluates discourse quality in Widow's Bay (Apple TV+) discussion threads — classifying posts as **analysis**, **reaction**, or **prediction**.

---

## Evaluation Report Summary

| Model | Accuracy |
|---|---|
| Zero-shot baseline (Groq llama-3.3-70b-versatile) | **90.9%** |
| Fine-tuned DistilBERT | **72.7%** |
| Δ | −18.2 points (fine-tuning regression) |

Both models evaluated on the same locked test set of 33 examples (70/15/15 stratified split from 215 total labeled examples).

---

## Community Choice

**Widow's Bay (Apple TV+) — r/television and r/WidowsBay finale discussion threads**

Widow's Bay is a horror-comedy limited series whose audience skews toward craft-aware viewers who discuss structure, tone, and character explicitly. The Season 1 finale (Episode 10, aired June 2026) generated simultaneous reaction posts, multi-paragraph structural analysis of the show's comedy/horror balance, and a dense cluster of Season 2 prediction posts driven by the renewal announcement and the 8-bell cliffhanger. All three label types appear in roughly equal volume in the same thread window — no hunting for underrepresented labels was needed. The show's mythology-heavy finale also produces prediction posts that include analytical reasoning, making it an ideal stress test for the analysis/prediction boundary that is the hardest one in the taxonomy.

---

## Label Taxonomy

### `analysis`
A post that evaluates a show's writing, direction, performances, or themes using specific, verifiable references to the show's text — scenes, dialogue, characters, structure — rather than just delivering a verdict.

**Example 1:**
> "The show works because the horror and the community are inseparable. The island's horrors don't exist despite the town's warmth — they exist because of it. The pact is what makes Widow's Bay Widow's Bay."

**Example 2:**
> "The Church Bells effectively being dinner bells was maybe my favourite punchline of the season. It recontextualizes every scene in the first episode where they ring."

---

### `reaction`
An immediate emotional or evaluative response to a specific episode or moment, with little to no supporting reasoning — the post is expressing a feeling or verdict rather than making an argument.

**Example 1:**
> "I shrieked with laughter when Dale told everyone to run. Then was immediately silenced with a gasp of horror when Ruth was shot. Man this episode was so fucking good."

**Example 2:**
> "I can't believe they stuck the landing this good."

---

### `prediction`
A speculative claim about what will happen in a future episode or season, stated as a forecast — the primary content is the prediction itself, not evaluation of something that already aired.

**Example 1:**
> "Prediction for season 2: Tom Wyck Bechir Patricia and maybe Dale will be the new sacrifice committee. Tom will also be focused on keeping anyone else especially Bechir and Wyck from finding out about Evan."

**Example 2:**
> "Eight more sacrifices huh? The next season is going to be about whether Tom becomes the new offering facilitator or tries to find another way out."

---

## Data Collection

**Source:** r/WidowsBay Episode 10 finale discussion thread and r/television Widow's Bay threads, collected in June 2026 within two weeks of the season finale airing.

**Labeling process:** 215 posts were collected manually from Reddit discussion threads, then labeled using the definitions in `planning.md`. Ambiguous cases were logged in the `notes` column of the CSV as they were encountered. No AI pre-labeling was used; all labels are manually assigned.

**Label distribution:**

| Label      | Count | % |
|------------|-------|---|
| reaction   | 78    | 36.3% |
| prediction | 70    | 32.6% |
| analysis   | 67    | 31.2% |
| **Total**  | **215** | |

No single label exceeds 70% of the dataset.

**5 difficult-to-label examples:**

1. **Post:** "I have a theory that Evan is immune to being sacrificed. It wouldn't make sense for the entity to kill Warren's relative. I think if Tom tells Evan it gives him the choice." — **Considered:** `analysis` vs `prediction` — **Decision:** `prediction`. The reasoning exists to support a forecast; removing the speculative claim leaves nothing complete.

2. **Post:** "I wonder if the reason no one knows about the council anymore is because they were tricked into becoming the previous sacrifice. Like Howard the Coward figured out a way to tie the people controlling the lever to the track themselves." — **Considered:** `analysis` vs `prediction` — **Decision:** `prediction`. Speculates about an unexplained mechanism as a forward-looking explanation, not evaluation of depicted events.

3. **Post:** "I loved the last bit though where you see Evan learning his lesson: he sticks his head into the room visibly considers going farther in and then is like nah actually the wiser choice is to get the fuck out of here and turns right around." — **Considered:** `reaction` vs `analysis` — **Decision:** `analysis`. Names a specific scene and explicitly interprets its significance — evaluative substance even in casual register.

4. **Post:** "Matthew Rhys wins my Emmy. No matter who else wins. That speech where he teased his wife. Brutal." — **Considered:** `reaction` vs `analysis` — **Decision:** `reaction`. Naming a scene and calling it "brutal" is an emotional verdict; no argument about why the craft choice works.

5. **Post:** "Oh I LOVE that Dale of all people is going to end the season as the person keeping most of the island's secrets" — **Considered:** `reaction` vs `analysis` — **Decision:** `reaction`. Expressing delight at an observation is not the same as evaluating whether the writing choice is good or explaining why it works.

---

## Fine-Tuning Pipeline

**Base model:** `distilbert-base-uncased` (HuggingFace)
**Training platform:** Google Colab (T4 GPU, free tier)
**Training split:** 150 train / 32 validation / 33 test (70/15/15 stratified)

**Hyperparameters:**
- Epochs: 3
- Learning rate: 2e-5
- Batch size: 16
- Weight decay: 0.01
- Warmup steps: 50
- Best model selection: by validation accuracy across epochs

**Key training decision:** The default 3-epoch setting was kept rather than increasing to 5 because with only 150 training examples, additional epochs increase overfitting risk while providing limited signal for a model of this size. Validation accuracy plateaued rather than continued increasing — the performance ceiling was data quantity, not training duration.

---

## Baseline

**Approach:** Groq's `llama-3.3-70b-versatile` was prompted zero-shot with the three label definitions copied verbatim from `planning.md`, one example post per label, and the explicit decision rules used during annotation. The model was instructed to output exactly one of the three label words with no explanation. Temperature was set to 0. All 33 test responses were parseable (0 unparseable).

**Prompt used:**
```
You are classifying posts from r/WidowsBay (the subreddit for the Apple TV+ show
Widow's Bay) into exactly one of three categories.

analysis — A post that evaluates the show's writing, direction, performances, or
themes using specific, verifiable references to the text — not just a verdict or
general praise/criticism.
Example: "The shelter was never a shelter — it was the holding cell for the
sacrifices. The 'how to subdue someone' poster and the 'check for weapons' sign
make that completely clear."

reaction — An immediate emotional response to a specific plot event or moment,
with little to no evaluative reasoning behind it.
Example: "WE KNEW IT WAS COMING AND IM STILL YELLING NO NO NO AT RUTH"

prediction — A speculative claim about what will happen in a future episode or
season, stated as a forecast rather than a response to something that already happened.
Example: "I'm guessing next season is going to be about them trying to avoid
sacrificing anyone for as long as possible while they investigate the demon and
try to undo the 300 year old deal."

Rules:
- If a post is primarily a forecast about future events, label it "prediction"
  even if it includes supporting reasoning.
- If a post evaluates something that already aired with specific references,
  label it "analysis."
- If a post is an emotional response with little evaluative content, label it "reaction."
- Choose exactly one label. Do not explain your reasoning.
- Respond with ONLY one of these three words, lowercase, no punctuation, nothing
  else: analysis, reaction, prediction
```

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|---|---|
| Zero-shot baseline (Groq llama-3.3-70b-versatile) | **90.9%** |
| Fine-tuned DistilBERT | **72.7%** |

### Per-Class Metrics — Fine-Tuned Model

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 0.583 | 0.700 | 0.636 | 10 |
| reaction | 1.000 | 0.667 | 0.800 | 12 |
| prediction | 0.692 | 0.818 | 0.750 | 11 |
| **macro avg** | **0.758** | **0.728** | **0.729** | **33** |

### Per-Class Metrics — Baseline

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 1.000 | 0.800 | 0.889 | 10 |
| reaction | 0.846 | 0.917 | 0.880 | 12 |
| prediction | 0.917 | 1.000 | 0.957 | 11 |
| **macro avg** | **0.921** | **0.906** | **0.909** | **33** |

The baseline outperforms the fine-tuned model on every label. The largest gap is on `analysis` (F1: 0.636 fine-tuned vs. 0.889 baseline). The fine-tuned model's one notable strength is reaction precision (1.000) — it never predicted `reaction` when the true label was something else — but its recall for reaction is only 0.667, meaning it missed a third of actual reaction posts.

### Confusion Matrix — Fine-Tuned Model

Rows = true label, columns = predicted label.

|  | → analysis | → reaction | → prediction |
|---|---|---|---|
| **analysis** | **7** | 0 | 3 |
| **reaction** | 3 | **8** | 1 |
| **prediction** | 2 | 0 | **9** |

Key pattern: off-diagonal errors are almost entirely concentrated in the analysis/prediction column. No examples were confused between `reaction` and `prediction` in either direction — that boundary is clean. The model struggles specifically with distinguishing `analysis` from `prediction`: 3 true analysis posts were predicted as prediction, and 2 true prediction posts were predicted as analysis.

See also: [`confusion_matrix.png`](confusion_matrix.png)

### Wrong Predictions — Error Analysis

**Error 1**
- **Post:** "My favorite show since Severance S1 by a wide margin."
- **True:** reaction | **Predicted:** analysis | **Confidence:** 0.34
- **Analysis:** A 9-word sentence with zero evaluative reasoning and no specific reference to the show's content, predicted as `analysis` — the most structurally demanding label. This reveals what the model actually learned: not the presence or absence of specific textual evidence (the intended boundary), but something closer to "does this post seem like it was written by someone who has opinions about TV." Short, casual declaratives triggered the wrong class. The low confidence confirms the model has no reliable signal here — it made an uncertain wrong choice, not a confident one.

**Error 2**
- **Post:** "My prediction: the show uses its second season to transform Tom from someone trying to escape the island's nature to someone who has accepted it and is trying to weaponize it. A villain origin story dressed as a comedy."
- **True:** prediction | **Predicted:** analysis | **Confidence:** 0.37
- **Analysis:** The post begins with the explicit phrase "My prediction:" and proceeds to forecast a future character arc. The model classified it as `analysis`. This directly tests the hardest edge case from `planning.md`: prediction posts that include structured reasoning. The decision rule states that reasoning in service of a forecast should be labeled `prediction`; the model learned the opposite — reasoning signals `analysis` regardless of whether it evaluates something past or forecasts something future. 150 training examples was not enough for DistilBERT to learn the temporal dimension of the taxonomy; it learned to associate structured text with `analysis` as a proxy.

**Error 3**
- **Post:** "So wait was the implication that Ruth just got grazed on the ear? I'm very confused on whether she is dead or not."
- **True:** reaction | **Predicted:** prediction | **Confidence:** 0.36
- **Analysis:** This is a confused immediate response to something that just aired — a `reaction` by the annotation rule — but its phrasing is forward-oriented ("whether she is dead or not") in a way that superficially resembles a forecast. The model's mistake is defensible at the surface level. However, the decision rule is clear: if the primary content is a response to something that just happened on screen, it's `reaction` even if the confusion generates apparent speculation. This error points to a data problem as much as a model problem: the training set likely contains too few examples of reaction posts phrased in a future-oriented way, leaving the model without signal for this sub-type.

### Sample Classifications

Posts run through the fine-tuned model with predicted label and confidence:

| Post (truncated) | True | Predicted | Confidence | Correct? |
|---|---|---|---|---|
| "Me sitting with my mouth open like Dale" | reaction | reaction | 0.37 | ✅ |
| "Many true laugh out loud moments. The episode when Rosemary did the family tree..." | analysis | analysis | 0.36 | ✅ |
| "I wonder if we'll get to see the island residents choose the victims next season..." | prediction | prediction | 0.40 | ✅ |
| "My prediction: the show uses its second season to transform Tom..." | prediction | analysis | 0.37 | ❌ |
| "My favorite show since Severance S1 by a wide margin." | reaction | analysis | 0.34 | ❌ |

**Why the second correct prediction is reasonable:** The Rosemary/family-tree post names a specific episode sequence, evaluates it with a concrete time estimate ("5-10 minutes of non-stop laughter"), and credits a specific performance ("facial expressions by Matthew Rhys are masterful"). It checks every box in the `analysis` definition — specific reference, evaluative claim, no future forecast — and the model correctly identifies it. That said, the confidence of 0.36 is notably low for a clear-cut case, which reflects the model's general uncertainty across all predictions and supports the finding that it did not learn confident, reliable decision boundaries.

---

## Reflection: What the Model Captured vs. What Was Intended

The intended classification axis was **epistemic mode**: is this post evaluating something that happened (analysis), responding emotionally to something that happened (reaction), or forecasting something that hasn't happened (prediction)?

What the model appears to have captured instead is a **text structure proxy**: longer, more complex posts with multi-clause sentences → analysis; short, high-affect posts → reaction; posts containing future-tense markers → prediction. This proxy works most of the time (hence 72.7% accuracy) but breaks down precisely where the taxonomy is most meaningful — at the analysis/prediction boundary, where both labels can involve structured multi-sentence text.

The specific failure pattern: 5 of 9 wrong predictions involve the model predicting `analysis` when the true label is `reaction` or `prediction`. The model over-predicts analysis, treating text complexity as sufficient evidence for the label. It never once predicted `reaction` for a true `prediction` post or vice versa — the temporal boundary between reaction and prediction was cleanly learned. The boundary that failed was analysis vs. prediction, exactly as anticipated in `planning.md`.

The baseline (Groq llama-3.3-70b-versatile) did not make this mistake because its language understanding is rich enough to track the difference between evaluating past events and forecasting future ones. DistilBERT fine-tuned on 150 examples learned surface features instead of the semantic distinction.

To fix this: the training set would need more examples of prediction-with-reasoning (posts that look structurally like analysis but are forecasts) explicitly labeled as `prediction`, so the model sees enough evidence that text complexity alone doesn't determine the `analysis` label.

A secondary finding reinforces the diagnosis: all five confidence scores in the Sample Classifications table — both correct and incorrect predictions — fall between 0.34 and 0.40. A well-trained classifier would show high confidence on clear cases and low confidence on hard ones. This model shows uniformly low confidence everywhere. It is not confidently wrong on hard cases and confidently right on easy ones; it is uncertain across the board. This is consistent with a model that learned weak, noisy surface features rather than the underlying semantic distinction the taxonomy was designed to capture.

---

## Spec Reflection

**One way the spec helped:** The requirement to write the edge case decision rule before annotating 200 examples was the most valuable pre-annotation step. Without it, the analysis/prediction boundary (posts that use evidence to support a forecast) would have been applied inconsistently across the dataset, producing noise the model couldn't learn from. Writing "could you remove the speculative sentence and still have a complete evaluative post?" as a mechanical test forced a decision rule precise enough to apply consistently.

**One way implementation diverged:** The spec's community suggestion was r/television (general). In practice, collecting from a single show's subreddit (r/WidowsBay) at a specific moment (the Season 1 finale) produced a far more coherent dataset than pulling from the general subreddit would have. General r/television posts vary enormously in topic, length, and style — which introduces noise unrelated to the label taxonomy. The single-show approach meant all posts shared the same narrative context, making label boundaries cleaner and annotation faster. The tradeoff is reduced generalizability: this classifier is essentially a Widow's Bay classifier, not a general TV discourse classifier.

---

## AI Usage

**Instance 1 — Label stress-testing:**
Claude was given the three label definitions and the analysis/prediction edge case rule and asked to generate 8–10 posts deliberately sitting at label boundaries. Several generated examples were genuinely ambiguous, exposing a gap in the prediction definition: the original definition didn't specify how to handle posts that both cite past evidence and make a future claim. The decision rule ("could you remove the speculative sentence and still have a complete post?") was written in response to this exercise. The AI generated the stress-test cases; the rule was written independently.

**Instance 2 — Failure pattern analysis:**
After fine-tuning, all 9 wrong predictions were shared with Claude and it was asked to surface common themes across them. It identified that errors clustered around the analysis/prediction boundary and that the model appeared to use text complexity as a proxy for the `analysis` label. This hypothesis was verified by re-reading each error independently before writing the error analysis section — the LLM's pattern claim was a starting point, not the analysis. One suggested pattern ("the model struggles with short posts") was partially overridden: the shortest post in the error set ("My favorite show since Severance S1") was misclassified, but not all short posts were — the real pattern was text complexity signaling label, not length per se.

---

## Deployed Interface (Stretch Feature)

A Gradio interface is available in [`src/interface.py`](src/interface.py).

**To run:**
```bash
pip install gradio transformers torch
# Download the best checkpoint from Colab first
python src/interface.py --model-dir ./takemeter-model/checkpoint-best
```

The interface accepts any post text, runs it through the fine-tuned model, and displays the predicted label and per-class confidence scores.

---

## Repository Structure

```
.
├── README.md
├── planning.md
├── takemeter_colab.ipynb       # Fine-tuning notebook (run on Colab T4 GPU)
├── data/
│   └── labeled_dataset.csv     # 215 labeled examples
├── src/
│   ├── collect_data.py         # Reddit scraper (optional)
│   ├── prelabel.py             # AI-assisted pre-labeling helper
│   ├── baseline.py             # Groq zero-shot baseline runner
│   └── interface.py            # Gradio deployed interface (stretch feature)
├── evaluation_results.json     # Downloaded from Colab after training
└── confusion_matrix.png        # Downloaded from Colab after training
```

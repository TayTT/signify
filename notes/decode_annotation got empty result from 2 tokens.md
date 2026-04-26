Sure. Let me build up from scratch because CTC has a few moving parts that need to click together.

---

### What CTC is doing during inference

Your LSTM processes a video sequence of T frames and produces a probability distribution over the vocabulary at **every single frame**. So if you have 200 frames and a vocab of 500 tokens, the output is a `(200, 500)` tensor — at each timestep the model is saying "here's how likely each token is right now".

The problem is: a video of someone signing "HEUTE REGEN" might be 200 frames long, but the annotation is only 2 glosses. You need a way to collapse 200 frame-level predictions into 2 tokens. That's what CTC does.

---

### Point 2 — blank token = 0 = `<PAD>`

CTC introduces a special **blank token** whose meaning is "nothing is being signed right now, or this frame is between glosses". The model is expected to output blank for most frames, and only emit an actual gloss token at the frames where that gloss is peaking.

In your code, blank is hardcoded to index 0. Your vocab also assigns index 0 to `<PAD>`. These are the same slot. At every frame, the model picks whichever token has the highest logit — and if it picks 0, that means "blank/nothing here".

When the aug model is in blank collapse, it's picking 0 at nearly every single frame across the entire sequence.

---

### Point 3 — what greedy decoding does with `[0, 0, 0, 2, 0, 0, 3, 0, 0]`

The greedy decoder in `_ctc_greedy_decode` does two things in order:

**Step 1 — collapse consecutive duplicates:**
`[0, 0, 0, 2, 0, 0, 3, 0, 0]` → `[0, 2, 0, 3, 0]`

**Step 2 — remove all blanks (token 0):**
`[0, 2, 0, 3, 0]` → `[2, 3]`

So if the model outputs mostly blank across 200 frames but happens to emit token 2 somewhere around frame 50 and token 3 somewhere around frame 150 — which it does because every target it was trained on starts with 2 and ends with 3 — you get exactly the `[2, 3]` pattern you see in the debug output. The model learned the mandatory boundary tokens but nothing in between.

---

### Point 4 — why `[2, 3]` then produces an empty string

`decode_annotation` receives `[2, 3]` and loops over them. For each token it looks up the text, then hits this filter:

```python
if token_text not in ('<PAD>', '<UNK>', '<SOS>', '<EOS>'):
    tokens.append(token_text)
```

Token 2 is `<SOS>` → skipped. Token 3 is `<EOS>` → skipped. Nothing gets appended. `' '.join([])` is an empty string. The debug print fires because of the empty result check just below that.

---

### Putting it together visually

```
Frame:    1    2    3  ...  50  ...  150  ...  200
Output:   0    0    0  ...   2  ...    3  ...    0   (mostly blank, SOS and EOS somewhere)

After collapse duplicates:  0, 2, 0, 3, 0
After strip blank (0):      2, 3
After filter SOS/EOS:       (empty)
```

Compare to what a working prediction looks like — the original model on a good frame:

```
Frame:    1  ...  20  ...  80  ...  140  ...  200
Output:   0  ...   2  ...  46  ...   89  ...    3   (blank, then SOS, then HEUTE, then REGEN, then EOS)

After collapse duplicates:  0, 2, 0, 46, 0, 89, 0, 3
After strip blank:          2, 46, 89, 3
After filter SOS/EOS:       HEUTE REGEN
```

The aug model never got stable enough training to push actual gloss tokens (46, 89, etc.) above blank in those middle frames.
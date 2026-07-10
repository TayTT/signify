Let me look at the actual code to give you a precise answer rather than guessing.Now I have the exact picture. Let me check the CTC setup too.Now I have everything. I need to correct my earlier analysis — I was wrong about what tokens 2 and 3 are. Let me give you the precise explanation.

---

### Correction first

I said tokens 2 and 3 were "almost certainly `__ON__` and `__OFF__`". That was wrong. Looking at the actual vocab construction in `preprocess_jsons.py` lines 88–91:

```python
self.vocab['<PAD>'] = 0
self.vocab['<UNK>'] = 1
self.vocab['<SOS>'] = 2
self.vocab['<EOS>'] = 3
```

Tokens 2 and 3 are `<SOS>` and `<EOS>`. Glosses including `__ON__` and `__OFF__` get indices 4+. This actually makes the situation more interesting.

---

### What's actually happening, step by step

**1. Targets are wrapped with SOS/EOS before going into CTC**

In `encode_annotation` (line 119):
```python
return [self.vocab['<SOS>']] + encoded + [self.vocab['<EOS>']]
```

So every target sequence fed to `F.ctc_loss` looks like `[2, gloss_a, gloss_b, ..., 3]`. CTC is not a seq2seq model — it doesn't conceptually need start/end tokens, but they're baked into every target here regardless. The model therefore has to learn to emit 2 and 3 as part of every valid alignment.

**2. CTC blank = 0, same as `<PAD>`**

In `_ctc_greedy_decode` (line 219) and `F.ctc_loss` (line 154), blank is hardcoded to 0, which is also `<PAD>`. The model output at any given frame timestep is one of: blank/PAD (0), or a gloss token (2+). When the model is collapsing to blank, it's outputting 0 for the vast majority of timesteps.

**3. What CTC greedy decoding produces for the aug model**

After collapsing consecutive duplicates and stripping blanks, only `[2, 3]` survive. The model is emitting token 0 (blank) almost everywhere, with maybe a single emission of 2 and 3 somewhere in the sequence — enough to "satisfy" the alignment loosely, but nothing else.

**4. Why `decode_annotation` then produces empty string**

In `decode_annotation` (line 140):
```python
if token_text not in ('<PAD>', '<UNK>', '<SOS>', '<EOS>'):
    tokens.append(token_text)
```

SOS and EOS are explicitly filtered out. So `[2, 3]` → empty list → empty string → the debug print fires.

**5. Why the original model doesn't have this problem as badly**

The original model (no aug, lower lr=1.2e-4) had a more stable training run. It learned to emit gloss tokens 4+ within the CTC alignment in addition to the forced SOS/EOS boundaries. The aug model with lr=1e-3 never stabilized enough to commit to interior gloss tokens — the repeated gradient explosions you saw in the loss curve kept resetting whatever interior-gloss patterns it was starting to learn.

---

### The underlying design issue worth noting

Having SOS/EOS inside CTC targets is redundant at best and slightly harmful at worst — CTC's job is purely to align frames to a flat label sequence, and the SOS/EOS tokens add two mandatory emissions to every alignment without conveying sign language content. They take up "alignment budget" from actual glosses, especially on short sequences. It doesn't break anything since both models use the same setup, but it's worth cleaning up before Stage 2 (mBART) anyway, since the gloss→text translation shouldn't receive SOS/EOS artifacts from Stage 1.
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
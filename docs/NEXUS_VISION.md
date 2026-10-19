# NEXUS Vision

NEXUS Vision is the model NEXUS Live uses to see your screen. It looks at a
screenshot and a request ("open Notepad and type hello") and answers with the
next mouse or keyboard step as JSON. It runs on your own computer, so there is
no per-token cost and screenshots never leave the machine.

It is a LoRA adapter on [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
(Apache-2.0). Training a vision model from nothing would take a data centre;
fine-tuning a small open one on NEXUS Live's own job takes a GPU and an hour or two.
What it learns:

- **NEXUS Live's exact prompts and answer format.** Every training example is the
  same conversation NEXUS Live has with the model at run time
  (`nexus/live/data.py`, `to_messages`), so a fine-tuned model answers in the shape
  NEXUS Live can act on, instead of drifting into prose or the wrong coordinates.
- **Finding things on a Windows screen.** OS-Atlas has hundreds of thousands of
  real desktop screenshots with every element's box. Training uses a few thousand
  Windows ones.
- **Your apps and habits**, from your own demonstrations in teach mode.

## 1. Record your own examples (optional, recommended)

```
nexus live --teach
```

Type the request you are about to carry out, then do it yourself the way you want
NEXUS to: click, type, press keys, scroll. Ctrl+Alt+Space pauses, Ctrl+Alt+Q
finishes. Each step is saved with the screenshot from just before it, in
`~/.nexus/live/teach`.

Before a session is saved you are shown everything you typed, and you can leave any
of it out. Nothing is recorded in password managers, banking, payment and terminal
windows (the same blocked list NEXUS Live uses), in teach mode's own console, or in
a window whose title cannot be read, whether it is in front or under a click.
Wherever such a window shows on the screen, it is blacked out in the screenshots,
whole, together with its program's other windows (its pop-ups, lists and dialogs);
only the window you are working in, in front, shows over it, since any other window
on top could be see-through (an overlay, a screen recorder's frame). A screenshot
taken when the windows on the screen could not be read is not used. All of this is
on Windows and Linux; a Mac cannot say where its windows are, so close private
windows there before teaching. Do not type passwords while teaching anyway.

The Windows key, Caps Lock, Print Screen and the volume keys are not saved as steps,
and characters made with AltGr count as typing. A step taken so soon after the one
before that no screenshot shows its effect, a click with Ctrl or Shift held, or a
combination NEXUS Live refuses to press (Ctrl+Esc, Ctrl+Shift+Esc) is not an example
of its own, but the examples after it still list it among the steps already taken.
A key whose release teach mode never heard (Win+L and Ctrl+Alt+Delete let go of the
keys on another screen) counts as up again once the keyboard says it is.

A few dozen sessions of the things you actually do makes a real difference.

## 2. Train it

Every way runs the same thing (`nexus vision build`): fetch the OS-Atlas
screenshots, add your recordings if you give them, fine-tune, test the result
against the base model on examples it never saw, and upload it to a **private**
model repo on your Hugging Face account.

You need a Hugging Face account and a token that can write
(https://huggingface.co/settings/tokens, New token, type Write).

### a. Kaggle (free, recommended)

Kaggle gives free GPU time every week (your quota is on your Kaggle settings page),
and a notebook saved with "Save & Run All" keeps running after you close the tab.

1. On kaggle.com, verify your phone number once (Settings). It unlocks GPUs and
   internet access in notebooks.
2. Create > New Notebook, then File > Import Notebook, and import
   `notebooks/train_nexus_vision.ipynb` from this repo (the GitHub link to it).
3. In the right panel: Accelerator **GPU T4 x2**, Internet **On**.
4. Add-ons > Secrets: add `HF_TOKEN` with your token, and tick it for this notebook.
5. In the settings cell, put your user name in `REPO`. Until the NEXUS Live branch
   is merged, set `REF = "claude/nexus-live"`.
6. Save Version > **Save & Run All (Commit)** > Save. Close the tab.

It takes a few hours. The model lands in your Hugging Face repo, and a copy sits in
the notebook's Output tab in case the upload fails.

The free T4 has no bfloat16 in hardware, so training runs in float16. If that
overflows (the training or held-out loss turns into NaN), it starts again in
float32 by itself: slower, but it holds.

While it runs, `run.log` in your model repo shows how far it is, and is updated
every ten minutes; a failure is written there with its reason.

### b. Colab (free)

Open `notebooks/train_nexus_vision.ipynb` in Google Colab, Runtime > Change runtime
type > T4 GPU, add `HF_TOKEN` under the key icon (Secrets), fill in the settings
cell and Runtime > Run all. Colab stops when you close the tab, so keep it open.

### c. Your own graphics card (free)

On Windows, first install the CUDA build of PyTorch from
https://pytorch.org/get-started/locally/ (pip's own torch on Windows has no CUDA and
would train on the processor), then:

```
pip install -e .[vision]
nexus vision build --push YOUR_NAME/nexus-vision
```

On a card under 12 GB the base model is loaded in 4 bits. A 4 GB laptop card (an
RTX 3050) can do it with `--width 768 --batch-size 1 --grad-accum 16 --limit 1500`,
slowly: leave it overnight. If you train at another width, NEXUS Live reads it from
the model and shrinks the screen to match.

### d. Hugging Face Jobs (paid)

Jobs bill your Hugging Face account by the minute and need a positive credit
balance (https://huggingface.co/settings/billing). An A10G costs about $1 an hour;
a run with the defaults takes about an hour and a half. The script stops itself
after 3 hours whatever happens.

```
hf auth login
hf jobs uv run tools/nexus_vision_job.py -- --push YOUR_NAME/nexus-vision
```

The script installs NEXUS from the `main` branch. Until NEXUS Live is merged, change
`@main` in its first lines to `@claude/nexus-live`.

### Your recordings in the cloud

Kaggle, Colab and Jobs train on your teach recordings only if you upload them,
which makes a private dataset repo:

```
nexus vision share-teach YOUR_NAME/nexus-teach
```

Then set `TEACH = "YOUR_NAME/nexus-teach"` in the notebook (or pass
`--teach YOUR_NAME/nexus-teach`). Look through `~/.nexus/live/teach/images` first if
anything private was on your screen. A repo of that name that already exists and is
public is refused, and so is a public model repo for `--push`.

### One step at a time

```
nexus vision fetch --system windows --limit 4000 --out data/vision
nexus vision train --data data/vision --data ~/.nexus/live/teach --out nexus_vision
nexus vision eval --data data/vision --model nexus_vision --compare Qwen/Qwen3-VL-2B-Instruct
```

Each takes `--help`.

## 3. Use it

```
nexus vision pull YOUR_NAME/nexus-vision
nexus live --check
nexus live --voice
```

`pull` puts it in `~/.nexus/live/nexus_vision` (or `NEXUS_LIVE_MODEL_DIR`, or
`--to`), where NEXUS Live's default backend loads it, replacing whatever model was
there. It downloads into a folder beside that one, and only a complete download takes
the old model's place: if the download stops, running `pull` again carries on where it
was, and if the new model cannot be moved in, the old one is put back. A folder that
has other things in it and no model is refused rather than emptied. For an adapter
it also downloads the base model it runs on (about 4.3 GB, once). After that it works
with no internet. On a graphics card under 8 GB it is loaded in 4 bits.

## How good is it?

`eval.json` and the README in your model repo compare NEXUS Vision with the base
model on held-out examples:

- **Steps right**: the same action as recorded and, for a click, inside the
  element's box (OS-Atlas) or within 3% of the screen of where you clicked (teach).
- **Invalid answers**: answers NEXUS Live could not act on. These are what make a
  request stop halfway, so this number matters as much as the first.
- **Clicks inside the box** and **median click distance**: how well it finds things.

The held-out examples are kept out of training by screenshot and by teach session,
so a screenshot the model learned from is never used to grade it.

`eval_details.json` has every graded answer: what the model said, how many tokens it
generated and what they were (special tokens included), and the error if the
question failed. The first three invalid answers of each model are also in
`run.log`.

## Files

What `nexus vision train` writes, and NEXUS Live loads:

| File | What it is |
|---|---|
| `adapter_config.json`, `adapter_model.safetensors` | the LoRA adapter |
| processor and tokenizer files | how screenshots and text are turned into tokens |
| `nexus_live.json` | the model card: the prompt version it learned, the position format, the screenshot width |
| `training.json` | what it was trained on and how |
| `eval.json`, `README.md` | the comparison with the base model (`nexus vision build`) |
| `eval_details.json` | every graded answer, for finding out why one was wrong |
| `run.log` | the run's progress, and the reason if it failed (`nexus vision build --push`) |

If NEXUS Live's prompts change, `PROMPT_VERSION` in `nexus/live/model.py` goes up and
NEXUS Live refuses a model trained on the old ones, with a message saying to train
again, rather than running it on prompts it never saw. Your recordings carry over:
the prompts around each one are rebuilt at training time, so one made under older
prompts is still used when its step passes today's checks (a typed text with a line
break in it, which the first prompts allowed, does not).

## Running it through llama.cpp instead

The local backend runs the model with transformers. To run it in llama.cpp's
`llama-server` (faster on a CPU), merge it and convert it to GGUF:

```
nexus vision merge ~/.nexus/live/nexus_vision nexus_vision_merged
python D:\unsloth\llama.cpp\convert_hf_to_gguf.py nexus_vision_merged --outfile nexus-vision.gguf
python D:\unsloth\llama.cpp\convert_hf_to_gguf.py nexus_vision_merged --mmproj --outfile mmproj-nexus-vision.gguf
llama-server -m nexus-vision.gguf --mmproj mmproj-nexus-vision.gguf --port 8080 -ngl 99
```

Then set `NEXUS_LIVE_BACKEND=llamacpp`.

## Data and licences

- Base model: Qwen/Qwen3-VL-2B-Instruct, Apache-2.0.
- OS-Atlas desktop grounding data: OS-Copilot/OS-Atlas-data, Apache-2.0. Its
  Windows screenshots sit in a 71 GB zip split in four parts;
  `nexus/live/atlas.py` reads the zip's table of contents from the end of the last
  part and fetches only the chosen screenshots with ranged reads, so a few thousand
  of them come to a few GB.
- Your teach recordings are yours. They stay in `~/.nexus/live/teach` unless you
  upload them with `nexus vision share-teach`, which makes a private dataset repo.

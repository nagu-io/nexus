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
windows (the same blocked list NEXUS Live uses). Do not type passwords while
teaching anyway.

A few dozen sessions of the things you actually do makes a real difference.

## 2. Train it

All three ways run the same thing (`nexus vision build`): fetch the OS-Atlas
screenshots, add your recordings if you give them, fine-tune, test the result
against the base model on examples it never saw, and upload it to a **private**
model repo on your Hugging Face account.

You need a Hugging Face account and a token that can write
(https://huggingface.co/settings/tokens).

### a. Hugging Face Jobs (paid, simplest)

Jobs bill your Hugging Face account by the minute and need a positive credit
balance (https://huggingface.co/settings/billing). An A10G costs about $1 an hour;
a run with the defaults takes about an hour and a half, so expect $1.50 to $3. The
script stops itself after 3 hours whatever happens.

```
hf auth login
hf jobs uv run tools/nexus_vision_job.py -- --push YOUR_NAME/nexus-vision
```

To add your recordings, upload them first (they stay private):

```
nexus vision share-teach YOUR_NAME/nexus-teach
hf jobs uv run tools/nexus_vision_job.py -- --push YOUR_NAME/nexus-vision --teach YOUR_NAME/nexus-teach
```

### b. Colab or Kaggle (free)

Open `notebooks/nexus_vision_colab.ipynb` in Google Colab (or Kaggle), choose a T4
GPU, set your repo name in the settings cell and run the cells in order. With 2000
OS-Atlas elements it takes about two hours on a free T4.

### c. Your own graphics card

```
pip install -e .[vision]
nexus vision build --push YOUR_NAME/nexus-vision
```

On a card under 12 GB the base model is loaded in 4 bits. A 4 GB laptop card (an
RTX 3050) can do it with `--width 960 --batch-size 1 --grad-accum 16 --limit 1500`,
slowly: plan for an evening. If you train at another width, NEXUS Live reads it
from the model and shrinks the screen to match.

The steps can also be run one at a time:

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

`pull` puts it in `~/.nexus/live/nexus_vision` (or `NEXUS_LIVE_MODEL_DIR`), where
NEXUS Live's default backend loads it. On a graphics card under 8 GB it is loaded in
4 bits. After the one download it works with no internet.

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

## Files

What `nexus vision train` writes, and NEXUS Live loads:

| File | What it is |
|---|---|
| `adapter_config.json`, `adapter_model.safetensors` | the LoRA adapter |
| processor and tokenizer files | how screenshots and text are turned into tokens |
| `nexus_live.json` | the model card: the prompt version it learned, the position format, the screenshot width |
| `training.json` | what it was trained on and how |
| `eval.json`, `README.md` | the comparison with the base model (`nexus vision build`) |

If NEXUS Live's prompts change, `PROMPT_VERSION` in `nexus/live/model.py` goes up and
NEXUS Live refuses a model trained on the old ones, with a message saying to train
again, rather than running it on prompts it never saw.

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

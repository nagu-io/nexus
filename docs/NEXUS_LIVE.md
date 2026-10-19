# NEXUS Live

NEXUS Live sees your screen, listens to you, talks back, and uses the mouse and
keyboard for you. Everything runs on your own computer: the vision model (your own
NEXUS Vision, see [NEXUS_VISION.md](NEXUS_VISION.md)), speech to text
(faster-whisper) and speech (the system's own voice). Nothing goes to a cloud model,
so there is no per-token cost and your screen stays on your machine.

## Set it up (Windows)

```
pip install -e .[live,vision]
nexus vision pull YOUR_NAME/nexus-vision
nexus live --check
```

`--check` looks at this computer, the vision model, the Python packages and the
screen, and says what is missing and how to fix it.

## Use it

| Command | What it does |
|---|---|
| `nexus live` | type requests in the console |
| `nexus live --voice` | press Ctrl+Alt+Space, speak, and hear the answer |
| `nexus live --watch` | also keep an eye on the screen and speak up when something needs you (an error, a finished download); it never acts on its own |
| `nexus live --goal "open notepad"` | do one request and exit |
| `nexus live --dry-run` | say what it would do, and touch nothing |
| `nexus live --teach` | do tasks yourself and save them as training examples |

Keys, which work whichever window is in front:

| Keys | |
|---|---|
| Ctrl+Alt+Space | talk (press again when you are done, or just stop talking) |
| Ctrl+Alt+Y / Ctrl+Alt+N | yes / no when it asks before a step |
| Ctrl+Alt+Q | stop what it is doing |
| mouse into a screen corner | emergency stop in the middle of a step |

## How it works

For each request it loops: take a screenshot, ask the vision model for the next
step, check the step, ask you, do it, look again. It stops when the model says the
request is done or answered, when you say no, when you press stop, or after 8 steps
(`NEXUS_LIVE_MAX_STEPS`), and it asks you to carry on rather than wandering.

## What keeps it safe

- **It asks before every step** by default: the pointer moves to where it would
  click, it says what it is about to do, and nothing happens until you press
  Ctrl+Alt+Y. No answer within 30 seconds counts as no. With
  `NEXUS_LIVE_CONFIRM=risky` clicks and scrolling go ahead and only typing and keys
  are asked about.
- **Windows it never acts in**: password managers, Windows Security, banking,
  PayPal, UPI and wallets, PowerShell, Command Prompt and terminals, the Registry
  Editor, Group Policy, Task Scheduler and UAC prompts. Matched as whole words in
  the title of the window in front; add your own with `NEXUS_LIVE_BLOCKED_WINDOWS`.
  `NEXUS_LIVE_ALLOWED_WINDOWS` turns this around: it then acts only in the windows
  you list.
- **Keys it never presses**: the Windows key (it opens the Run box), Ctrl+Alt+Delete
  and Ctrl+Shift+Esc.
- **It checks the window again right before acting.** If a pop-up or another window
  came forward after the screenshot, it looks again instead of clicking into it.
- **It never clicks off the screenshot.** A position outside it is treated as a
  misread, not clamped to the nearest edge.
- **The model is told** never to type passwords, one-time codes, card or bank
  details, or to confirm payments, and to hand those parts back to you.
- **Watch mode only speaks.** It has no hands.

## Settings

Put these in `.env` or the environment.

| Variable | Default | |
|---|---|---|
| `NEXUS_LIVE_BACKEND` | `local` | `local`: NEXUS Vision in this process. `llamacpp`: a GGUF in llama-server. `ollama`: a model in Ollama |
| `NEXUS_LIVE_MODEL_DIR` | `~/.nexus/live/nexus_vision` | where NEXUS Vision is, for `local` |
| `NEXUS_LIVE_LOAD_4BIT` | `auto` | `auto` loads it in 4 bits on graphics cards under 8 GB; `on`, `off` |
| `NEXUS_LIVE_LLAMACPP_URL` | `http://127.0.0.1:8080` | for `llamacpp` |
| `NEXUS_LIVE_MODEL` | `auto` | the Ollama model for `ollama`; `auto` picks one for this computer |
| `NEXUS_LIVE_CONFIRM` | `all` | `all` or `risky` |
| `NEXUS_LIVE_MAX_STEPS` | `8` | steps for one request before it stops and asks |
| `NEXUS_LIVE_SCREEN_WIDTH` | `1280` | screenshots are shrunk to this; by default the width NEXUS Vision was trained at |
| `NEXUS_LIVE_BLOCKED_WINDOWS` | | more window words it never acts in, comma separated |
| `NEXUS_LIVE_ALLOWED_WINDOWS` | | if set, the only windows it acts in |
| `NEXUS_LIVE_HOTKEY_TALK`, `_STOP`, `_YES`, `_NO` | as above | in pynput's format, such as `<ctrl>+<alt>+q` |
| `NEXUS_LIVE_WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, `medium` |
| `NEXUS_LIVE_SPEAK` | `true` | speak answers aloud |
| `NEXUS_LIVE_TIMEOUT` | `240` | seconds to wait for the model |
| `NEXUS_LIVE_CONFIRM_TIMEOUT` | `30` | seconds to wait for your yes |
| `NEXUS_LIVE_WATCH_INTERVAL` | `10` | seconds between looks in watch mode |

## Without NEXUS Vision

Until you have trained your own model, NEXUS Live runs on a stock vision model in
Ollama:

```
ollama pull qwen3-vl:4b
nexus live --model qwen3-vl:4b
```

With `NEXUS_LIVE_BACKEND=ollama` and no model named, it picks one for this
computer's memory (`nexus live --check` says which).

## Limits

- One screen: it sees and acts on the main monitor.
- Drags and drawing are not among its steps; teach mode records the press as a click.
- A 2B model on a laptop takes a few seconds per step, more on a CPU.
- On Linux, pyautogui needs tkinter (`sudo apt install python3-tk`).

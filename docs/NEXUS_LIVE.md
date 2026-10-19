# NEXUS Live

NEXUS Live sees your screen, listens to you, talks back, and uses the mouse and
keyboard for you. Everything runs on your own computer: the vision model (your own
NEXUS Vision, see [NEXUS_VISION.md](NEXUS_VISION.md)), speech to text
(faster-whisper) and speech (the system's own voice). Nothing goes to a cloud model,
so there is no per-token cost and your screen stays on your machine.

## Set it up (Windows)

Install the CUDA build of PyTorch first, from https://pytorch.org/get-started/locally/
(pip's own torch on Windows has no CUDA, and NEXUS Vision would run on the processor,
slowly). Then:

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
| `nexus live` | type a request in the console, then click the window it is for |
| `nexus live --voice` | press Ctrl+Alt+Space, speak, and hear the answer |
| `nexus live --watch` | also keep an eye on the screen and speak up when something needs you (an error, a finished download); it never acts on its own |
| `nexus live --goal "open notepad"` | do one request and exit (click the window it is for, too) |
| `nexus live --dry-run` | say what it would do, and touch nothing |
| `nexus live --teach` | do tasks yourself and save them as training examples |

Keys, which work whichever window is in front. On Windows the window being worked
on never gets them, and their Alt is the left Alt: on keyboards where the right Alt
is AltGr, AltGr+Q still types @. Keys a program types, NEXUS Live's own typing
included, never count as these keys.

| Keys | |
|---|---|
| Ctrl+Alt+Space | talk (press again when you are done, or just stop talking) |
| Ctrl+Alt+Y / Ctrl+Alt+N | yes / no when it asks before a step |
| Ctrl+Alt+Q | stop what it is doing (a question waiting for your answer gets a no at once); pressed while you are still talking, it drops what you said |
| mouse into a screen corner | emergency stop in the middle of a step |

NEXUS Live never types or clicks in its own console. When you type a request there,
the console is in front, so it asks you to click the window the request is for and
starts when that window comes forward.

## How it works

For each request it loops: take a screenshot, ask the vision model for the next
step, check the step, ask you, do it, look again. It stops when the model says the
request is done or answered, when you say no, when you press stop, or after 8 steps
(`NEXUS_LIVE_MAX_STEPS`), and it asks you to carry on rather than wandering. If
something fails (the model runs out of memory, say), that request ends and NEXUS
Live waits for the next one. In voice mode the same goes for the microphone and
speech to text, and speech to text moves to the processor if the graphics card fails
it.

## What keeps it safe

- **It asks before every step** by default: the pointer moves to where it would
  click, and nothing happens until you press Ctrl+Alt+Y. No answer within 30
  seconds counts as no. With `NEXUS_LIVE_CONFIRM=risky` clicks and scrolling go
  ahead and only typing and keys are asked about.
- **The question is built from what NEXUS Live checked, not from what the model
  says.** It names the action, the window it lands in, and for typing every
  character it would type and how many. The model's own words about the step are
  not read out while you decide: text on the screen can steer a model, and it could
  then describe a harmful step as a safe one. Nor are its words on the way (while
  it waits, or before a click `risky` does not ask about), which you would hear
  just before the next question: you hear only its final answer.
- **Typing is one line at a time.** Enter and Tab are steps of their own, each
  asked about, so a yes to some text is never also a yes to pressing Enter. Text in
  other languages is typed as characters on Windows; on a Mac or Linux it goes
  through the clipboard, which keeps that text afterwards.
- **It waits for your hands.** After your yes it waits until Ctrl and Alt are
  released, so the step is not mixed with the keys of the yes. If you press a
  modifier key while it types, or another window comes to the front, it stops
  typing, so the rest of the text never lands where you did not agree to.
- **Windows it never acts in**: password managers, Windows Security, banking,
  PayPal, UPI and wallets, PowerShell, Command Prompt, Windows Terminal, Git Bash,
  MSYS2, Cygwin and Anaconda Prompt, the Registry Editor, Group Policy, Task
  Scheduler, Task Manager and UAC prompts, matched as whole words in window titles;
  and the Run box, the Start menu, search and Task View, matched as whole titles.
  It checks both the window in front and, for a click or scroll, the window under
  the pointer, since a click lands in whatever window is there; when it cannot tell
  which window that is, it asks even with `risky`. Add your own words with
  `NEXUS_LIVE_BLOCKED_WINDOWS`. `NEXUS_LIVE_ALLOWED_WINDOWS` turns this around: it
  then acts only in the windows you list.
- **Keys it never presses**: the Windows key and Ctrl+Esc (they open the Start menu
  and the Run box), Ctrl+Alt+Delete, Ctrl+Shift+Esc, and its own hotkeys.
- **It checks again right before acting.** If another window came forward, a
  different window is now under the click, or the screen near the click has changed
  (a dialog closed, a list scrolled) since the screenshot the model saw, it looks
  again instead of acting on what is no longer there, and says so (once a request).
  A place that never stops changing, such as a playing video, is told apart by two
  more looks a moment apart: still changing, it goes ahead; settled into something
  new, it looks again.
- **It never clicks off the screenshot.** A position outside it is treated as a
  misread, not clamped to the nearest edge.
- **The model is told** never to type passwords, one-time codes, card or bank
  details, or to confirm payments, and to hand those parts back to you.
- **Watch mode only speaks.** It has no hands, and it pauses while a request runs.

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
- On a Mac it cannot tell which window is under a click, so with `risky` it asks
  about clicks too. It reads the window in front through AppleScript, which takes a
  moment each time, and typing reads it before every letter, so typing is slow there.
- On Linux, pyautogui needs tkinter (`sudo apt install python3-tk`).

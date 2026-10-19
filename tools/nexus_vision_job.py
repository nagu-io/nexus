# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "nexus-ai[vision] @ git+https://github.com/nagu-io/nexus@main",
# ]
#
# [tool.hf-jobs]
# flavor = "a10g-small"
# timeout = "3h"
# secrets = ["HF_TOKEN"]
# ///
"""Train NEXUS Vision on a Hugging Face GPU, and upload it to your account.

    hf jobs uv run tools/nexus_vision_job.py -- --push YOUR_NAME/nexus-vision

Runs nexus.live.build: fetches OS-Atlas Windows screenshots, fine-tunes
Qwen3-VL-2B-Instruct, tests it against the base model on held-out examples, and
uploads it to a private model repo. Every option of nexus vision build works after the
``--``, such as ``--teach YOUR_NAME/nexus-teach`` to add your own recordings.

Jobs bill your Hugging Face account by the minute while this runs (an A10G is
about $1 an hour; the run takes about an hour and a half), and the timeout above
stops it after 3 hours whatever happens.
"""

from nexus.live.build import main

if __name__ == "__main__":
    raise SystemExit(main())

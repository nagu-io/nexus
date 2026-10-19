"""Tests for nexus.live.finetune.AnswerCollator with a stand-in processor: which tokens the model learns from."""

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None

HEADER = [7, 8, 9]  # "<|im_start|>assistant\n"
END = 5  # "<|im_end|>"
PAD = 0


class FakeTokenizer:
    pad_token_id = PAD

    def encode(self, text, add_special_tokens=False):
        return list(HEADER)

    def convert_tokens_to_ids(self, token):
        return END


class FakeProcessor:
    """Returns the token rows it is given, padded on the right, as the real processor would."""

    def __init__(self, rows):
        self.rows = rows
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, conversations, tokenize, return_dict, return_tensors, **padding):
        import torch

        width = max(len(row) for row in self.rows)
        ids = torch.tensor([row + [PAD] * (width - len(row)) for row in self.rows])
        mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in self.rows])
        return {"input_ids": ids, "attention_mask": mask}


@unittest.skipUnless(HAS_TORCH, "needs torch")
class CollatorTests(unittest.TestCase):
    def labels(self, rows):
        from nexus.live.data import Example
        from nexus.live.finetune import AnswerCollator

        example = Example(id="e", source="t", group="g", image="i.png", request="r", target={"action": "done", "say": "x"})
        collator = AnswerCollator(FakeProcessor(rows), 1280)
        with mock.patch("nexus.live.finetune.load_image", return_value="IMAGE"):
            batch = collator([(Path("."), example)] * len(rows))
        return batch["labels"].tolist()

    def test_only_the_answer_and_its_end_token_are_learned(self):
        row = [1, 2, 3] + HEADER + [41, 42, 43] + [END, 99]
        self.assertEqual(self.labels([row])[0], [-100] * 6 + [41, 42, 43, END, -100])

    def test_the_last_answer_header_counts(self):
        row = HEADER + [11, END] + [3] + HEADER + [21, 22, END]
        self.assertEqual(self.labels([row])[0], [-100] * 9 + [21, 22, END])

    def test_padding_is_never_learned(self):
        short = [1] + HEADER + [41, END]
        long = [1, 2, 3, 4] + HEADER + [41, 42, END]
        labels = self.labels([short, long])
        self.assertEqual(labels[0], [-100] * 4 + [41, END] + [-100] * 4)
        self.assertEqual(labels[1], [-100] * 7 + [41, 42, END])

    def test_no_whole_answer_teaches_nothing(self):
        self.assertEqual(self.labels([[1, 2, 3, 41, END]])[0], [-100] * 5)  # no header
        self.assertEqual(self.labels([[1] + HEADER + [41, 42]])[0], [-100] * 6)  # cut off before the end token
        self.assertEqual(self.labels([[END, 1] + HEADER + [41]])[0], [-100] * 6)  # an end token only before it


if __name__ == "__main__":
    unittest.main()

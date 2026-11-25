from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "distillation" / "prepare_sft_dataset.py"


class ArabicSftPreparationTests(unittest.TestCase):
    def test_deduplicates_and_splits_valid_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "teacher.jsonl"
            output = base / "prepared"
            response_a = "شرح عربي صحيح ومفصل. " * 8
            response_b = "مثال عربي آخر للتدريب. " * 8
            rows = [
                {"prompt": "اشرح النحو", "response": response_a, "area": "النحو"},
                {"prompt": "اشرح النحو", "response": response_a, "area": "النحو"},
                {"prompt": "اشرح الصرف", "response": response_b, "area": "الصرف"},
                {"prompt": "", "response": response_b},
                {"prompt": "سؤال غير مكتمل", "response": "قصير"},
            ]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--input", str(source),
                    "--output-dir", str(output),
                    "--validation-ratio", "0.5",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads((output / "dataset_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["accepted"], 2)
            self.assertEqual(report["training"], 1)
            self.assertEqual(report["validation"], 1)
            self.assertEqual(report["rejected_or_duplicate"], 3)
            self.assertIn('"accepted": 2', result.stdout)

    def test_rejects_invalid_validation_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "teacher.jsonl"
            source.write_text("", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT),
                    "--input", str(source),
                    "--output-dir", str(Path(tmp) / "out"),
                    "--validation-ratio", "0.8",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("validation-ratio", result.stderr)


if __name__ == "__main__":
    unittest.main()

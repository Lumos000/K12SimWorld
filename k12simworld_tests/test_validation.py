import unittest

from k12simworld.validation import validate_document


class ValidationTest(unittest.TestCase):
    def test_blocks_networked_browser_code(self):
        report = validate_document("<canvas></canvas><script>fetch('https://x')</script>", "p5js")
        self.assertFalse(report.valid)
        self.assertTrue(any("network" in error for error in report.errors))

    def test_blocks_remote_script(self):
        report = validate_document(
            '<canvas></canvas><script src="https://cdn.example/p5.js"></script>', "p5js"
        )
        self.assertFalse(report.valid)

    def test_blocks_non_whitelisted_manim_import(self):
        report = validate_document(
            "import sys\nfrom manim import *\nclass GeneratedScene(Scene):\n    pass\n", "manim"
        )
        self.assertFalse(report.valid)

    def test_accepts_minimal_manim_scene(self):
        report = validate_document("from manim import *\nclass GeneratedScene(Scene):\n    def construct(self):\n        pass\n", "manim")
        self.assertTrue(report.valid, report.errors)

    def test_blocks_unsafe_manim_import(self):
        report = validate_document("import subprocess\nclass GeneratedScene(Scene):\n    pass\n", "manim")
        self.assertFalse(report.valid)


if __name__ == "__main__":
    unittest.main()

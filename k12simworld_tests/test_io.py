import unittest

from k12simworld.io import safe_artifact_name


class IoTest(unittest.TestCase):
    def test_untrusted_id_cannot_escape_output_directory(self):
        result = safe_artifact_name("../../outside")
        self.assertNotIn("/", result)
        self.assertNotIn("..", result)


if __name__ == "__main__":
    unittest.main()

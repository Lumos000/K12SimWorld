import unittest

from k12simworld.curation import assign_knowledge_disjoint_splits, build_benchmark
from k12simworld.models import K12Problem


def problem(pid, subject="physics", kp="kp"):
    return K12Problem.from_record(
        {
            "hash_id": pid,
            "question": "物体移动时状态如何变化？",
            "subject": f"{subject}-g9",
            "type": "选择题",
            "knowledge_point": [kp],
            "format_answer": {"format_solution": ["step"], "ground_truth": ["A"]},
            "dynamic_suitability": True,
        }
    )


class CurationTest(unittest.TestCase):
    def test_selection_is_deterministic(self):
        items = [problem(f"p{i}") for i in range(10)] + [problem(f"m{i}", "math") for i in range(4)]
        first = build_benchmark(items, physics_target=5, extension_target=2, expert_target=2)
        second = build_benchmark(items, physics_target=5, extension_target=2, expert_target=2)
        self.assertEqual([p.problem_id for p in first.selected], [p.problem_id for p in second.selected])

    def test_shared_knowledge_point_never_crosses_split(self):
        items = [problem("p1", kp="same"), problem("p2", kp="same"), problem("p3", kp="other")]
        splits = assign_knowledge_disjoint_splits(items)
        self.assertEqual(splits["p1"], splits["p2"])


if __name__ == "__main__":
    unittest.main()

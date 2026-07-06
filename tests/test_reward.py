import unittest

from student_kit.reward import reward, score_svg


GOOD_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
<rect x="32" y="32" width="192" height="192" rx="28" fill="#F8E9C8"/>
<circle cx="128" cy="128" r="72" fill="#2D6CDF"/>
<path d="M88 146 C112 104 144 104 168 146" fill="none" stroke="#32A852" stroke-width="12" stroke-linecap="round"/>
</svg>"""


class RewardTest(unittest.TestCase):
    def test_good_svg_scores_high(self):
        result = score_svg(GOOD_SVG, prompt="rounded-square badge with a blue circle and green leaf curve")
        self.assertGreater(result["score"], 0.75)
        self.assertTrue(result["details"]["validity"]["xml_valid"])

    def test_bad_svg_scores_low(self):
        result = score_svg("not svg at all", prompt="blue circle")
        self.assertLess(result["score"], 0.2)

    def test_reward_returns_float(self):
        self.assertIsInstance(reward(GOOD_SVG), float)


if __name__ == "__main__":
    unittest.main()


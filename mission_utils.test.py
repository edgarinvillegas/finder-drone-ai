# Python code to demonstrate working of unittest
import unittest
from utils import mission_from_str

class TestMission(unittest.TestCase):
    def test_mission_from_str(self):
        str_mission = 'ffbbblr'
        output = mission_from_str(str_mission)
        self.assertEqual(output, [
            {'direction': 'forward', 'steps': 2},
            {'direction': 'back', 'steps': 3},
            {'direction': 'left', 'steps': 1},
            {'direction': 'right', 'steps': 1},
        ])

if __name__ == '__main__':
    unittest.main()
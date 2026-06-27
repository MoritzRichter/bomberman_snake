"""Tests für evolution/sensors.py — MoveHelper"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evolution'))

import unittest
from game.logic import GameLogic
from game.constants import Direction, FieldType, FIELDSIZE
from sensors import MoveHelper
from profiles import get_profile, profile_input_size, limit_profile


def make_game(level=1) -> GameLogic:
    """Frisches Spiel ohne zufällige Elemente — Essen danach manuell setzen."""
    g = GameLogic(level)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# can_move — Wand-Checks
# ─────────────────────────────────────────────────────────────────────────────

class TestCanMoveWalls(unittest.TestCase):

    def test_forward_into_wall_is_false(self):
        """Schlange läuft geradeaus gegen eine Wand."""
        g = make_game()
        # Level 1: Außenwand bei x=0. Schlange mit Kopf bei (1,5) schaut LEFT → nächste Zelle (0,5) ist Wand
        g.snake = [(1, 5), (2, 5), (3, 5)]
        g.direction = Direction.LEFT
        g.food_pos = (5, 5)
        h = MoveHelper(g)
        self.assertFalse(h.can_move(MoveHelper.FORWARD))

    def test_forward_into_free_is_true(self):
        """Schlange läuft in freie Zelle vorwärts."""
        g = make_game()
        g.snake = [(5, 5), (4, 5), (3, 5)]
        g.direction = Direction.RIGHT
        g.food_pos = (8, 8)
        h = MoveHelper(g)
        self.assertTrue(h.can_move(MoveHelper.FORWARD))

    def test_left_into_wall_is_false(self):
        """Links liegt eine Wand."""
        g = make_game()
        # Schaut RIGHT, links wäre UP → Kopf bei (5,8), Wand bei (5,9)
        g.snake = [(5, 8), (4, 8), (3, 8)]
        g.direction = Direction.RIGHT
        g.food_pos = (2, 2)
        h = MoveHelper(g)
        # links von RIGHT ist UP → (5,9) = Außenwand in Level 1
        self.assertFalse(h.can_move(MoveHelper.LEFT))

    def test_right_into_wall_is_false(self):
        """Rechts liegt eine Wand."""
        g = make_game()
        # Schaut RIGHT, rechts wäre DOWN → Kopf bei (5,1), Wand bei (5,0)
        g.snake = [(5, 1), (4, 1), (3, 1)]
        g.direction = Direction.RIGHT
        g.food_pos = (2, 2)
        h = MoveHelper(g)
        # rechts von RIGHT ist DOWN → (5,0) = Außenwand in Level 1
        self.assertFalse(h.can_move(MoveHelper.RIGHT))


# ─────────────────────────────────────────────────────────────────────────────
# can_move — Body-Collision-Checks (der gefixte Bug)
# ─────────────────────────────────────────────────────────────────────────────

class TestCanMoveBodyCollision(unittest.TestCase):

    def test_forward_into_own_body_is_false(self):
        """Schlange läuft geradeaus in ihr eigenes Körpersegment."""
        g = make_game()
        # Kopf (5,5) schaut RIGHT, Körpersegment bei (6,5) direkt voraus
        g.snake = [(5, 5), (6, 5), (7, 5)]
        g.direction = Direction.RIGHT
        g.food_pos = (2, 2)
        h = MoveHelper(g)
        self.assertFalse(h.can_move(MoveHelper.FORWARD))

    def test_left_into_own_body_is_false(self):
        """Schlange dreht links in ein eigenes Körpersegment — der gefixte Bug."""
        g = make_game()
        # Kopf (5,5) schaut RIGHT → links ist UP → Segment bei (5,6) ist Körper
        g.snake = [(5, 5), (5, 6), (5, 7)]
        g.direction = Direction.RIGHT
        g.food_pos = (2, 2)
        h = MoveHelper(g)
        self.assertFalse(h.can_move(MoveHelper.LEFT))

    def test_right_into_own_body_is_false(self):
        """Schlange dreht rechts in ein eigenes Körpersegment — der gefixte Bug."""
        g = make_game()
        # Kopf (5,5) schaut RIGHT → rechts ist DOWN → Segment bei (5,4) ist Körper
        g.snake = [(5, 5), (5, 4), (5, 3)]
        g.direction = Direction.RIGHT
        g.food_pos = (2, 2)
        h = MoveHelper(g)
        self.assertFalse(h.can_move(MoveHelper.RIGHT))

    def test_left_free_when_body_is_forward(self):
        """Körpersegment liegt vorwärts, nicht links — links soll True sein."""
        g = make_game()
        # Kopf (5,5) schaut RIGHT → Körper bei (6,5) vorwärts, links (5,6) frei
        g.snake = [(5, 5), (6, 5), (7, 5)]
        g.direction = Direction.RIGHT
        g.food_pos = (2, 2)
        h = MoveHelper(g)
        # Früher (Bug): can_move(LEFT) gab False, weil Körper vorwärts lag
        # Jetzt korrekt: links (5,6) ist frei → True
        self.assertTrue(h.can_move(MoveHelper.LEFT))


# ─────────────────────────────────────────────────────────────────────────────
# is_food
# ─────────────────────────────────────────────────────────────────────────────

class TestIsFood(unittest.TestCase):

    def test_food_forward(self):
        g = make_game()
        g.snake = [(5, 5), (4, 5), (3, 5)]
        g.direction = Direction.RIGHT
        g.food_pos = (8, 5)   # rechts vom Kopf = FORWARD
        h = MoveHelper(g)
        self.assertTrue(h.is_food(MoveHelper.FORWARD))
        self.assertFalse(h.is_food(MoveHelper.LEFT))
        self.assertFalse(h.is_food(MoveHelper.RIGHT))

    def test_food_left(self):
        g = make_game()
        g.snake = [(5, 5), (4, 5), (3, 5)]
        g.direction = Direction.RIGHT
        g.food_pos = (5, 8)   # über dem Kopf = LEFT wenn Richtung RIGHT
        h = MoveHelper(g)
        self.assertTrue(h.is_food(MoveHelper.LEFT))
        self.assertFalse(h.is_food(MoveHelper.FORWARD))

    def test_food_right(self):
        g = make_game()
        g.snake = [(5, 5), (4, 5), (3, 5)]
        g.direction = Direction.RIGHT
        g.food_pos = (5, 2)   # unter dem Kopf = RIGHT wenn Richtung RIGHT
        h = MoveHelper(g)
        self.assertTrue(h.is_food(MoveHelper.RIGHT))
        self.assertFalse(h.is_food(MoveHelper.FORWARD))


# ─────────────────────────────────────────────────────────────────────────────
# is_bomb
# ─────────────────────────────────────────────────────────────────────────────

class TestIsBomb(unittest.TestCase):

    def test_no_bomb_returns_false(self):
        g = make_game()
        g.snake = [(5, 5), (4, 5)]
        g.direction = Direction.RIGHT
        g.food_pos = (2, 2)
        g.bomb = False
        g.bomb_pos = None
        h = MoveHelper(g)
        self.assertFalse(h.is_bomb(MoveHelper.FORWARD))
        self.assertFalse(h.is_bomb(MoveHelper.LEFT))
        self.assertFalse(h.is_bomb(MoveHelper.RIGHT))

    def test_bomb_forward(self):
        g = make_game()
        g.snake = [(5, 5), (4, 5)]
        g.direction = Direction.RIGHT
        g.food_pos = (2, 2)
        g.bomb = True
        g.bomb_pos = (8, 5)   # vorwärts
        h = MoveHelper(g)
        self.assertTrue(h.is_bomb(MoveHelper.FORWARD))
        self.assertFalse(h.is_bomb(MoveHelper.LEFT))
        self.assertFalse(h.is_bomb(MoveHelper.RIGHT))

    def test_bomb_left(self):
        g = make_game()
        g.snake = [(5, 5), (4, 5)]
        g.direction = Direction.RIGHT
        g.food_pos = (2, 2)
        g.bomb = True
        g.bomb_pos = (5, 8)   # links (UP) wenn Richtung RIGHT
        h = MoveHelper(g)
        self.assertTrue(h.is_bomb(MoveHelper.LEFT))
        self.assertFalse(h.is_bomb(MoveHelper.FORWARD))


# ─────────────────────────────────────────────────────────────────────────────
# get_inputs
# ─────────────────────────────────────────────────────────────────────────────

class TestGetInputs(unittest.TestCase):

    def test_no_profile_returns_fourteen_values(self):
        """Without a profile, all 14 sensors are returned at weight 1.0."""
        g = make_game()
        g.snake = [(5, 5), (4, 5), (3, 5)]
        g.direction = Direction.RIGHT
        g.food_pos = (8, 5)
        g.bomb = False
        g.bomb_pos = None
        h = MoveHelper(g)
        self.assertEqual(len(h.get_inputs()), 14)

    def test_profile_input_count_matches_enabled_sensors(self):
        """get_inputs() length must equal the number of enabled sensors in the profile."""
        g = make_game()
        g.snake = [(5, 5), (4, 5), (3, 5)]
        g.direction = Direction.RIGHT
        g.food_pos = (8, 5)
        g.bomb = False
        g.bomb_pos = None
        for name in ("basic", "bomb_aware", "timer", "full"):
            profile = get_profile(name)
            h = MoveHelper(g, profile)
            self.assertEqual(len(h.get_inputs()), profile_input_size(profile), f"Profile '{name}' mismatch")

    def test_profile_basic_has_six_inputs(self):
        self.assertEqual(profile_input_size(get_profile("basic")), 6)

    def test_profile_bomb_aware_has_nine_inputs(self):
        self.assertEqual(profile_input_size(get_profile("bomb_aware")), 9)

    def test_profile_timer_has_twelve_inputs(self):
        self.assertEqual(profile_input_size(get_profile("timer")), 12)

    def test_profile_full_has_fourteen_inputs(self):
        self.assertEqual(profile_input_size(get_profile("full")), 14)

    def test_weighted_input_is_scaled(self):
        """A sensor with weight 2.0 should produce a value in [0.0, 2.0]."""
        g = make_game()
        g.snake = [(5, 5), (4, 5), (3, 5)]
        g.direction = Direction.RIGHT
        g.food_pos = (8, 5)
        g.bomb = True
        g.bomb_pos = (8, 5)
        g.bomb_timer = 1
        profile = get_profile("full")
        h = MoveHelper(g, profile)
        inputs = h.get_inputs()
        for val in inputs:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 2.0)  # max weight in profiles is 2.0

    def test_inputs_in_range(self):
        """All inputs must be in [0.0, 1.0]; can_move stays 0 or 1, food/bomb are continuous."""
        g = make_game()
        g.snake = [(5, 5), (4, 5), (3, 5)]
        g.direction = Direction.RIGHT
        g.food_pos = (8, 5)
        g.bomb = False
        g.bomb_pos = None
        h = MoveHelper(g, get_profile("bomb_aware"))
        inputs = h.get_inputs()
        # can_move sensors (indices 0-2) are still binary
        for val in inputs[:3]:
            self.assertIn(val, (0, 1))
        # food/bomb sensors (indices 3-8) are continuous 0.0–1.0
        for val in inputs[3:]:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# profiles — minimal + limit_profile
# ─────────────────────────────────────────────────────────────────────────────

class TestProfiles(unittest.TestCase):

    def test_minimal_profile_has_three_inputs(self):
        self.assertEqual(profile_input_size(get_profile("minimal")), 3)

    def test_limit_profile_reduces_count(self):
        full = get_profile("full")
        limited = limit_profile(full, 6)
        self.assertEqual(profile_input_size(limited), 6)

    def test_limit_profile_keeps_first_n_active(self):
        """Limiting the full profile to 3 should keep the first 3 sensors (can_move ×3)."""
        full    = get_profile("full")
        limited = limit_profile(full, 3)
        active  = [name for name, w in limited.items() if w > 0]
        self.assertEqual(active, ["can_move_forward", "can_move_left", "can_move_right"])

    def test_limit_profile_get_inputs_correct_length(self):
        g = make_game()
        g.snake = [(5, 5), (4, 5), (3, 5)]
        g.direction = Direction.RIGHT
        g.food_pos = (8, 5)
        g.bomb = False
        g.bomb_pos = None
        for n in (3, 6, 9, 12, 14):
            limited = limit_profile(get_profile("full"), n)
            h = MoveHelper(g, limited)
            self.assertEqual(len(h.get_inputs()), n, f"Expected {n} inputs")

    def test_limit_profile_raises_if_n_too_large(self):
        with self.assertRaises(ValueError):
            limit_profile(get_profile("minimal"), 10)


if __name__ == '__main__':
    unittest.main()

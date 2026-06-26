"""
Input profiles — select which sensors are active and how strongly they're weighted.

Weight = 0.0  → sensor is disabled (not passed to the network)
Weight = 1.0  → normal signal strength
Weight > 1.0  → amplified signal (network sees it as more important from the start)

Available sensors (14 total):
  can_move_forward / can_move_left / can_move_right
  is_food_forward  / is_food_left  / is_food_right
  is_bomb_forward  / is_bomb_left  / is_bomb_right
  food_timer       / bomb_timer    / explosion_active
  food_distance    / snake_length
"""

PROFILES: dict[str, dict[str, float]] = {

    # 3 inputs — absolute minimum: only collision awareness
    "minimal": {
        "can_move_forward" : 1.0,
        "can_move_left"    : 1.0,
        "can_move_right"   : 1.0,
        "is_food_forward"  : 0.0,
        "is_food_left"     : 0.0,
        "is_food_right"    : 0.0,
        "is_bomb_forward"  : 0.0,
        "is_bomb_left"     : 0.0,
        "is_bomb_right"    : 0.0,
        "food_timer"       : 0.0,
        "bomb_timer"       : 0.0,
        "explosion_active" : 0.0,
        "food_distance"    : 0.0,
        "snake_length"     : 0.0,
    },

    # 6 inputs — safe moves + food direction
    "basic": {
        "can_move_forward" : 1.0,
        "can_move_left"    : 1.0,
        "can_move_right"   : 1.0,
        "is_food_forward"  : 1.0,
        "is_food_left"     : 1.0,
        "is_food_right"    : 1.0,
        "is_bomb_forward"  : 0.0,
        "is_bomb_left"     : 0.0,
        "is_bomb_right"    : 0.0,
        "food_timer"       : 0.0,
        "bomb_timer"       : 0.0,
        "explosion_active" : 0.0,
        "food_distance"    : 0.0,
        "snake_length"     : 0.0,
    },

    # 9 inputs — adds directional bomb awareness
    "bomb_aware": {
        "can_move_forward" : 1.0,
        "can_move_left"    : 1.0,
        "can_move_right"   : 1.0,
        "is_food_forward"  : 1.0,
        "is_food_left"     : 1.0,
        "is_food_right"    : 1.0,
        "is_bomb_forward"  : 1.0,
        "is_bomb_left"     : 1.0,
        "is_bomb_right"    : 1.0,
        "food_timer"       : 0.0,
        "bomb_timer"       : 0.0,
        "explosion_active" : 0.0,
        "food_distance"    : 0.0,
        "snake_length"     : 0.0,
    },

    # 12 inputs — adds urgency timers (food_timer amplified to emphasise danger)
    "timer": {
        "can_move_forward" : 1.0,
        "can_move_left"    : 1.0,
        "can_move_right"   : 1.0,
        "is_food_forward"  : 1.0,
        "is_food_left"     : 1.0,
        "is_food_right"    : 1.0,
        "is_bomb_forward"  : 1.0,
        "is_bomb_left"     : 1.0,
        "is_bomb_right"    : 1.0,
        "food_timer"       : 2.0,
        "bomb_timer"       : 2.0,
        "explosion_active" : 1.0,
        "food_distance"    : 0.0,
        "snake_length"     : 0.0,
    },

    # 14 inputs — everything enabled
    "full": {
        "can_move_forward" : 1.0,
        "can_move_left"    : 1.0,
        "can_move_right"   : 1.0,
        "is_food_forward"  : 1.0,
        "is_food_left"     : 1.0,
        "is_food_right"    : 1.0,
        "is_bomb_forward"  : 1.0,
        "is_bomb_left"     : 1.0,
        "is_bomb_right"    : 1.0,
        "food_timer"       : 2.0,
        "bomb_timer"       : 2.0,
        "explosion_active" : 1.0,
        "food_distance"    : 1.0,
        "snake_length"     : 1.0,
    },
}


def get_profile(name: str) -> dict[str, float]:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Available: {list(PROFILES.keys())}")
    return PROFILES[name]


def profile_input_size(profile: dict[str, float]) -> int:
    """Number of active (weight > 0) sensors in this profile."""
    return sum(1 for w in profile.values() if w > 0)


def limit_profile(profile: dict[str, float], n: int) -> dict[str, float]:
    """
    Return a copy of the profile with only the first n active sensors enabled.
    Sensors beyond the limit are set to 0.0 (disabled).

    Example:
        limit_profile(get_profile("full"), 6)
        → keeps can_move ×3 + is_food ×3, disables everything else
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    result  = {}
    active  = 0
    for name, weight in profile.items():
        if weight > 0 and active < n:
            result[name] = weight
            active += 1
        else:
            result[name] = 0.0

    if active < n:
        raise ValueError(
            f"Profile only has {active} active sensors, cannot limit to {n}"
        )
    return result

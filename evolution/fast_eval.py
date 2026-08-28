"""
fast_eval.py — Numba-JIT accelerated brain evaluation.

Drop-in replacement for the inner game loop in _eval_brain().
Transparent fallback to Python path if numba is not installed.

Architecture
------------
* network_to_arrays(brain)   → CSR weight arrays (Python, called once per brain)
* profile_to_arrays(profile) → sensor weight/index arrays (Python, called once per profile)
* _activate()                → njit forward pass (CSR)
* _cross_field()             → njit explosion setter
* _place_food()              → njit random food placement
* _run_game()                → njit full game loop (sensors + network + game step)
* eval_brain_fast()          → Python entry point; returns same format as _eval_brain inner loop
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from numba import njit
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False

# ---------------------------------------------------------------------------
# Constants (mirror game/constants.py — must stay in sync)
# ---------------------------------------------------------------------------
_FS          = 10   # FIELDSIZE
_MAX_SNAKE   = 101  # max snake length (10×10+1 safety)
_FOOD_STEPS  = 6
_BOMB_STEPS  = 3
_EXPL_STEPS  = 2
_EXPL_REACH  = 3

# FieldType integers
_FREE    = 0
_WALL    = 1
_EXPLODED = 2

# Direction integers: LEFT=0  RIGHT=1  UP=2  DOWN=3
# DIR_DX[d], DIR_DY[d] = displacement
# TURN_LEFT[d]  = direction when turning left
# TURN_RIGHT[d] = direction when turning right
# OPPOSITE[d]   = reverse direction

# Precomputed level grids as int8 numpy arrays
def _build_level_grids():
    from game.levels import LEVELS
    return {k: np.array(v, dtype=np.int8) for k, v in LEVELS.items()}

_LEVEL_GRIDS = _build_level_grids()

# Ordered list of all 19 sensor names (must match SENSOR_FUNCS in sensors.py)
_SENSOR_NAMES = [
    "can_move_forward",  # 0
    "can_move_left",     # 1
    "can_move_right",    # 2
    "is_food_forward",   # 3
    "is_food_left",      # 4
    "is_food_right",     # 5
    "is_bomb_forward",   # 6
    "is_bomb_left",      # 7
    "is_bomb_right",     # 8
    "food_timer",        # 9
    "bomb_timer",        # 10
    "explosion_active",  # 11
    "food_distance",     # 12
    "snake_length",      # 13
    "body_forward",      # 14
    "body_left",         # 15
    "body_right",        # 16
    "is_food_backward",  # 17
    "is_bomb_backward",  # 18
]

# ---------------------------------------------------------------------------
# Python helper: convert NEAT Network → flat numpy arrays (CSR format)
# ---------------------------------------------------------------------------
def network_to_arrays(brain):
    """Convert a Network to numba-compatible arrays.

    Returns
    -------
    biases     : float64[n_nodes]
    squash_ids : int8[n_nodes]     0=sigmoid  1=tanh  2=relu  3=identity
    in_ptr     : int32[n_nodes+1]  CSR row-start pointer
    in_src     : int32[n_conns]    CSR source node indices
    in_wgt     : float64[n_conns]  CSR connection weights
    n_inputs   : int
    n_outputs  : int
    n_nodes    : int
    """
    try:
        from evolution.constants import ALL_ACTIVATIONS
    except ModuleNotFoundError:
        from constants import ALL_ACTIVATIONS
    squash_map = {fn: i for i, fn in enumerate(ALL_ACTIVATIONS)}

    n = len(brain.nodes)
    biases     = np.array([nd.bias for nd in brain.nodes], dtype=np.float64)
    squash_ids = np.array([squash_map.get(nd.squash, 0) for nd in brain.nodes],
                          dtype=np.int8)

    node_idx = {id(nd): i for i, nd in enumerate(brain.nodes)}

    # Build CSR: for each node j, list its incoming connections in order
    in_ptr = np.zeros(n + 1, dtype=np.int32)
    for j, nd in enumerate(brain.nodes):
        in_ptr[j + 1] = in_ptr[j] + len(nd.connections_in)

    total = int(in_ptr[n])
    in_src = np.zeros(total, dtype=np.int32)
    in_wgt = np.zeros(total, dtype=np.float64)
    for j, nd in enumerate(brain.nodes):
        base = int(in_ptr[j])
        for k, conn in enumerate(nd.connections_in):
            in_src[base + k] = node_idx[id(conn.from_node)]
            in_wgt[base + k] = conn.weight

    return biases, squash_ids, in_ptr, in_src, in_wgt, brain.input_size, brain.output_size, n


# Module-level caches — survive across calls within the same process/thread
_profile_arrays_cache: dict = {}   # id(profile) -> (sensor_weights, active_indices)
_score_params_cache:   dict = {}   # scoring_mode -> float64[6]


def profile_to_arrays(profile):
    """Convert a profile dict → sensor weight array + active-index array.

    active_indices is ordered by PROFILE insertion order (not SENSOR_FUNCS order)
    so that activations[ii] = sensors[active_indices[ii]] reproduces the same
    input vector as MoveHelper.get_inputs(), which iterates over the profile dict.

    Returns
    -------
    sensor_weights : float64[19]   weight per sensor in SENSOR_FUNCS order
                                   (used by _sensors_all to scale values)
    active_indices : int32[n_inputs]  SENSOR_FUNCS indices in PROFILE order
                                      (used to pack activations in network-input order)
    """
    name_to_idx = {name: i for i, name in enumerate(_SENSOR_NAMES)}
    weights = np.zeros(19, dtype=np.float64)
    active  = []
    for name, w in profile.items():
        if name == "_raw":
            continue
        idx = name_to_idx[name]
        weights[idx] = w
        if w > 0.0:
            active.append(idx)   # append in PROFILE order
    return weights, np.array(active, dtype=np.int32)


def score_params_from_config():
    """Pack current ScoreConfig into a float64[6] array for the JIT."""
    try:
        from evolution.constants import config as _cfg
    except ModuleNotFoundError:
        from constants import config as _cfg
    sc = _cfg.score
    return np.array([
        sc.points_towards_food,
        sc.points_against_food,
        sc.points_ate_food,
        sc.points_survived_tick,
        sc.points_bomb_exploded,
        sc.points_per_length_tick,
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Numba JIT functions
# ---------------------------------------------------------------------------
if _NUMBA_OK:

    @njit(cache=True)
    def _activate(activations, biases, squash_ids, in_ptr, in_src, in_wgt, n_inputs):
        """Forward pass over non-input nodes. Modifies `activations` in-place.
        Input activations (0..n_inputs-1) must already be set by the caller.
        """
        n = len(activations)
        for j in range(n_inputs, n):
            state = biases[j]
            for k in range(in_ptr[j], in_ptr[j + 1]):
                state += activations[in_src[k]] * in_wgt[k]
            sid = squash_ids[j]
            if sid == 0:    # sigmoid
                activations[j] = 1.0 / (1.0 + np.exp(-state))
            elif sid == 1:  # tanh
                activations[j] = np.tanh(state)
            elif sid == 2:  # relu
                activations[j] = max(0.0, state)
            else:           # identity
                activations[j] = state


    @njit(cache=True)
    def _cross_field(grid, cx, cy, field_val):
        """Set explosion pattern (+ shape) around (cx, cy). Stops at walls."""
        grid[cy, cx] = field_val
        arms_dx = (-1, 1, 0,  0)
        arms_dy = ( 0, 0, 1, -1)
        for a in range(4):
            dx = arms_dx[a]
            dy = arms_dy[a]
            for i in range(1, _EXPL_REACH + 1):
                nx = (cx + dx * i) % _FS
                ny = (cy + dy * i) % _FS
                if grid[ny, nx] == _WALL:
                    break
                grid[ny, nx] = field_val


    @njit(cache=True)
    def _place_food(grid, snake, snake_len):
        """Return (fx, fy) for a new food cell: free grid cell, not on snake."""
        while True:
            fx = np.random.randint(0, _FS)
            fy = np.random.randint(0, _FS)
            if grid[fy, fx] != _FREE:
                continue
            hit = False
            for i in range(snake_len):
                if snake[i, 0] == fx and snake[i, 1] == fy:
                    hit = True
                    break
            if not hit:
                return fx, fy


    @njit(cache=True)
    def _sensors_all(out, body_map, grid, snake, snake_len, direction,
                     food_x, food_y, bomb_x, bomb_y, bomb_active,
                     food_timer, bomb_timer, expl_active,
                     weights):
        """Fill `out` (19,) with weighted sensor values.
        `body_map` (10×10 int8) is built here for O(1) body lookups.
        Both arrays are pre-allocated by the caller to avoid per-turn allocation.
        """
        # --- Build 10×10 body presence map (segments 1..snake_len-1) ---
        for _y in range(_FS):
            for _x in range(_FS):
                body_map[_y, _x] = 0
        for i in range(1, snake_len):
            body_map[snake[i, 1], snake[i, 0]] = 1

        # Direction lookup tables  LEFT=0  RIGHT=1  UP=2  DOWN=3
        tl = (3, 2, 0, 1)   # TURN_LEFT
        tr = (2, 3, 1, 0)   # TURN_RIGHT
        dx = (-1, 1, 0,  0)
        dy = ( 0, 0, 1, -1)

        hx = snake[0, 0]
        hy = snake[0, 1]

        # --- 0-2: can_move_forward / left / right ---
        for rel in range(3):
            if   rel == 0: abs_d = direction
            elif rel == 1: abs_d = tl[direction]
            else:          abs_d = tr[direction]
            nx = (hx + dx[abs_d]) % _FS
            ny = (hy + dy[abs_d]) % _FS
            safe = (grid[ny, nx] == _FREE) and (body_map[ny, nx] == 0)
            out[rel] = 1.0 if safe else 0.0

        # --- 3-5, 17: food direction distances ---
        def _food_dist(abs_d):
            if   abs_d == 1: dist = food_x - hx if food_x > hx else 0
            elif abs_d == 0: dist = hx - food_x if food_x < hx else 0
            elif abs_d == 2: dist = food_y - hy if food_y > hy else 0
            else:            dist = hy - food_y if food_y < hy else 0
            return (1.0 - dist / _FS) if dist > 0 else 0.0

        out[3] = _food_dist(direction)
        out[4] = _food_dist(tl[direction])
        out[5] = _food_dist(tr[direction])

        # --- 6-8, 18: bomb direction distances ---
        def _bomb_dist(abs_d):
            if not bomb_active:
                return 0.0
            if   abs_d == 1: dist = bomb_x - hx if bomb_x > hx else 0
            elif abs_d == 0: dist = hx - bomb_x if bomb_x < hx else 0
            elif abs_d == 2: dist = bomb_y - hy if bomb_y > hy else 0
            else:            dist = hy - bomb_y if bomb_y < hy else 0
            return (1.0 - dist / _FS) if dist > 0 else 0.0

        out[6] = _bomb_dist(direction)
        out[7] = _bomb_dist(tl[direction])
        out[8] = _bomb_dist(tr[direction])

        # --- 9-13: timers + flags ---
        out[9]  = food_timer / _FOOD_STEPS
        out[10] = (bomb_timer / _BOMB_STEPS) if bomb_active else 0.0
        out[11] = 1.0 if expl_active else 0.0
        out[12] = (abs(food_x - hx) + abs(food_y - hy)) / 18.0
        out[13] = snake_len / 100.0

        # --- 14-16: body proximity ray cast — O(FIELDSIZE) using body_map ---
        def _body_prox(abs_d):
            bx = hx; by = hy
            for dist in range(1, _FS):
                bx = (bx + dx[abs_d]) % _FS
                by = (by + dy[abs_d]) % _FS
                if body_map[by, bx] != 0:
                    return 1.0 - dist / _FS
            return 0.0

        out[14] = _body_prox(direction)
        out[15] = _body_prox(tl[direction])
        out[16] = _body_prox(tr[direction])

        # --- 17-18: backward ---
        back = (1, 0, 3, 2)
        out[17] = _food_dist(back[direction])
        out[18] = _bomb_dist(back[direction])

        # Apply profile weights in-place
        for i in range(19):
            out[i] *= weights[i]


    @njit(cache=True)
    def _run_game(base_grid, biases, squash_ids, in_ptr, in_src, in_wgt,
                  n_inputs, n_outputs, n_nodes,
                  max_turns, lowest_score,
                  sensor_weights, active_indices,
                  score_params):
        """Run one full game episode.

        Returns (score, turns, food_eaten,
                 sc_survival, sc_towards, sc_against, sc_ate, sc_bomb)

        score_params: float64[6]
            [towards_food, against_food, ate_food, survived_tick,
             bomb_exploded, per_length_tick]
        """
        # Direction tables
        tl = (3, 2, 0, 1)   # TURN_LEFT
        tr = (2, 3, 1, 0)   # TURN_RIGHT
        op = (1, 0, 3, 2)   # OPPOSITE
        dx = (-1, 1, 0,  0)
        dy = ( 0, 0, 1, -1)

        # --- Init game state ---
        grid = base_grid.copy()

        snake     = np.zeros((_MAX_SNAKE, 2), dtype=np.int32)
        snake[0, 0] = 6;  snake[0, 1] = 6
        snake[1, 0] = 5;  snake[1, 1] = 6
        snake[2, 0] = 4;  snake[2, 1] = 6
        snake_len = 3
        direction = 1  # RIGHT

        food_x, food_y = _place_food(grid, snake, snake_len)

        bomb_x    = 0;  bomb_y    = 0
        bomb_act  = False;  bomb_timer = 0
        expl_x    = 0;  expl_y    = 0
        expl_act  = False;  expl_timer = 0
        food_timer = 0

        score       = 0.0
        sc_survival = 0.0
        sc_towards  = 0.0
        sc_against  = 0.0
        sc_ate      = 0.0
        sc_bomb     = 0.0
        food_eaten  = 0
        turns       = 0

        activations = np.zeros(n_nodes, dtype=np.float64)
        sensors_out = np.zeros(19, dtype=np.float64)
        body_map    = np.zeros((_FS, _FS), dtype=np.int8)
        n_active    = len(active_indices)

        while turns < max_turns:

            # --- Sensors (fills sensors_out in-place, no allocation) ---
            _sensors_all(
                sensors_out, body_map,
                grid, snake, snake_len, direction,
                food_x, food_y, bomb_x, bomb_y, bomb_act,
                food_timer, bomb_timer, expl_act,
                sensor_weights,
            )

            # Pack active sensors into network inputs
            for ii in range(n_active):
                activations[ii] = sensors_out[active_indices[ii]]

            # --- Network forward pass ---
            _activate(activations, biases, squash_ids, in_ptr, in_src, in_wgt, n_inputs)

            # Read outputs from tail of activations (outputs are always last)
            out0 = activations[n_nodes - n_outputs]      # turn_left
            out1 = activations[n_nodes - n_outputs + 1]  # turn_right

            # --- Direction decision ---
            if round(out0):
                new_dir     = tl[direction]
                food_sensor = sensors_out[4]   # is_food_left (weighted)
            elif round(out1):
                new_dir     = tr[direction]
                food_sensor = sensors_out[5]   # is_food_right (weighted)
            else:
                new_dir     = direction
                food_sensor = sensors_out[3]   # is_food_forward (weighted)

            # U-turns are ignored (game rejects them), keep current direction
            if new_dir == op[direction]:
                new_dir     = direction
                food_sensor = sensors_out[3]

            # --- Score food direction (before move) ---
            if food_sensor > 0.0:
                pts = score_params[0]   # points_towards_food
                score += pts;  sc_towards += pts
            else:
                pts = score_params[1]   # points_against_food
                score += pts;  sc_against += pts

            # Update direction (U-turn already rejected above)
            direction = new_dir

            # --- Move snake ---
            nx = (snake[0, 0] + dx[direction]) % _FS
            ny = (snake[0, 1] + dy[direction]) % _FS

            # Wall or exploded cell → game over
            if grid[ny, nx] != _FREE:
                turns += 1
                break

            # Self-collision
            hit = False
            for i in range(1, snake_len):
                if snake[i, 0] == nx and snake[i, 1] == ny:
                    hit = True
                    break
            if hit:
                turns += 1
                break

            ate_food = (nx == food_x and ny == food_y)

            # Shift snake body (insert new head)
            smax = min(snake_len, _MAX_SNAKE - 1)
            for i in range(smax, 0, -1):
                snake[i, 0] = snake[i - 1, 0]
                snake[i, 1] = snake[i - 1, 1]
            snake[0, 0] = nx;  snake[0, 1] = ny
            if ate_food:
                if snake_len < _MAX_SNAKE:
                    snake_len += 1
                food_x, food_y = _place_food(grid, snake, snake_len)
                food_timer = 0
            # (if not ate_food, tail is already overwritten by the shift above)

            turns += 1

            # --- Cut tail at explosion ---
            if expl_act:
                cut_at = -1
                for i in range(snake_len):
                    if grid[snake[i, 1], snake[i, 0]] == _EXPLODED:
                        cut_at = i
                        break
                if cut_at == 0:   # head in explosion
                    break
                elif cut_at > 0:
                    snake_len = cut_at

            # --- Tick explosion ---
            had_expl = expl_act
            if expl_act:
                expl_timer += 1
                if expl_timer >= _EXPL_STEPS:
                    expl_timer = 0;  expl_act = False
                    _cross_field(grid, expl_x, expl_y, _FREE)

            # --- Tick food (only if food was not just eaten) ---
            if not ate_food:
                food_timer += 1
                if food_timer >= _FOOD_STEPS:
                    bomb_x = food_x;  bomb_y = food_y
                    bomb_act = True;  bomb_timer = 0
                    food_x, food_y = _place_food(grid, snake, snake_len)
                    food_timer = 0

            # --- Tick bomb ---
            if bomb_act:
                bomb_timer += 1
                if bomb_timer >= _BOMB_STEPS:
                    bomb_act = False;  bomb_timer = 0
                    expl_x = bomb_x;  expl_y = bomb_y
                    expl_act = True;  expl_timer = 0
                    _cross_field(grid, expl_x, expl_y, _EXPLODED)

            # --- Score: food eaten ---
            if ate_food:
                pts = score_params[2]   # points_ate_food
                score += pts;  sc_ate += pts
                food_eaten += 1

            # --- Score: new explosion penalty ---
            if not had_expl and expl_act:
                pts = score_params[4]   # points_bomb_exploded
                score += pts;  sc_bomb += pts

            # --- Score: survival ---
            pts = score_params[3]       # points_survived_tick
            score += pts;  sc_survival += pts
            if score_params[5] != 0.0:
                pts = score_params[5] * snake_len
                score += pts;  sc_survival += pts

            # --- Cutoff: score too low ---
            if score < lowest_score:
                break

        return score, turns, food_eaten, sc_survival, sc_towards, sc_against, sc_ate, sc_bomb


# ---------------------------------------------------------------------------
# Python entry point
# ---------------------------------------------------------------------------

def eval_brain_fast(brain, levels_to_play, max_turns, lowest_score,
                    profile, scoring_mode):
    """Evaluate `brain` over the given levels using JIT acceleration.

    Returns the same `totals` dict that _eval_brain accumulates manually.
    Raises RuntimeError if numba is not available (caller should catch and fall back).
    """
    if not _NUMBA_OK:
        raise RuntimeError("numba not installed")

    try:
        from evolution.constants import apply_scoring_preset
    except ModuleNotFoundError:
        from constants import apply_scoring_preset
    apply_scoring_preset(scoring_mode)

    biases, squash_ids, in_ptr, in_src, in_wgt, n_inputs, n_outputs, n_nodes = \
        network_to_arrays(brain)

    pid = id(profile)
    if pid not in _profile_arrays_cache:
        _profile_arrays_cache[pid] = profile_to_arrays(profile)
    sensor_weights, active_indices = _profile_arrays_cache[pid]

    if scoring_mode not in _score_params_cache:
        _score_params_cache[scoring_mode] = score_params_from_config()
    sp = _score_params_cache[scoring_mode]

    totals = {k: 0.0 for k in ("score", "turns", "food_eaten",
                                "score_survival", "score_towards",
                                "score_against", "score_ate", "score_bomb")}

    for lvl in levels_to_play:
        base_grid = _LEVEL_GRIDS[lvl].copy()
        sc, tu, fe, ss, st, sa, sate, sbomb = _run_game(
            base_grid, biases, squash_ids, in_ptr, in_src, in_wgt,
            n_inputs, n_outputs, n_nodes,
            max_turns, lowest_score,
            sensor_weights, active_indices, sp,
        )
        totals["score"]          += sc
        totals["turns"]          += tu
        totals["food_eaten"]     += fe
        totals["score_survival"] += ss
        totals["score_towards"]  += st
        totals["score_against"]  += sa
        totals["score_ate"]      += sate
        totals["score_bomb"]     += sbomb

    return totals

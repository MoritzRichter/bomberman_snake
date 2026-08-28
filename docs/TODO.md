# Bomberman Snake — TODO

## Erledigt

### Bugs

- [x] **Body-Kollision in `can_move(LEFT/RIGHT)`** (`evolution/sensors.py`)
  - `abs_dir` wird jetzt korrekt für LEFT/RIGHT berechnet, bevor `_step()` aufgerufen wird

### Inputs

Alle 19 Sensoren sind in `sensors.py` + `SENSOR_FUNCS` + `profiles.py` implementiert.

- [x] **Food-Dringlichkeit** — `game.food_timer / FOOD_STEPS` (0.0–1.0)
- [x] **Bomb-Dringlichkeit** — `game.bomb_timer / BOMB_STEPS` (0.0–1.0)
- [x] **Explosion aktiv** — `1 if game.explosion else 0`
- [x] **Distanz zum Essen** — `manhattan(head, food_pos) / (2 * FIELDSIZE)` (0.0–1.0)
- [x] **Schlangenlänge** — `len(game.snake) / (FIELDSIZE * FIELDSIZE)` (0.0–1.0)
- [x] **Körpernähe** — Ray-Cast pro Achse (`body_forward` / `body_left` / `body_right`)
- [x] **Raw-Profil** — kompletter Spielzustand als 108-Werte-Vektor, ohne Handcrafting

Profile: `minimal` (3), `basic` (6), `bomb_aware` (12), `timer` (15), `full` (19), `raw` (108)

### Score-Tuning (`evolution/constants.py`)

- [x] `points_against_food` gesenkt — Bomben ausweichen wird nicht stark bestraft
- [x] `points_survived_tick` — kleiner Überlebens-Bonus pro Schritt
- [x] `points_bomb_exploded` — Schlange lernt aktiv, Bomben zu verhindern
- [x] Fünf umschaltbare Presets (`balanced` / `survival` / `food` / `length` / `in_game_score`)

### Selektion

- [x] **Tournament-Selection** — K zufällige Gehirne ziehen, besten als Elternteil wählen
- [x] Fünf Strategien in `selection.py`: `power`, `tournament`, `roulette`, `top_n`, `random`

### eternity.py / eternity_deep.py

- [x] Zwischenstand speichern (Checkpoint nach Bootstrap-Erfolg und jeder Evo-Verbesserung)
- [x] Temp-Elite-System (Kandidaten müssen sich über `TEMP_PROMOTE_AFTER` Runden behaupten)
- [x] GPU-Frage → kein Vorteil (variable Topologie, sequenzieller Game-Loop; CPU ist richtig)
- [x] Seed-Offspring-Fix (neue Gehirne via `getOffspring(seeds)` + `mutatePopulation` statt Random)
- [x] Stabiler Elite-Pool (nur via TempElite-Promotion austauschbar, außer in Generation 0)
- [x] Multi-Game-Evaluation (`EVAL_GAMES = 3`): Score über 3 Spiele mitteln
- [x] Level-Rotation (`LEVEL_ROTATION`): opt-in, ein Spiel pro Level → robustere Agenten
- [x] Numba-JIT-Pfad (`fast_eval.py`) mit automatischem Python-Fallback
- [x] Adaptive Mutationsstrategie (EXPLORE ↔ EXPLOIT) in `eternity_deep.py`

---

## Offen

- [ ] **Dynamisches Turn-Limit** — `max_turns = Basis + food_eaten × Bonus` statt fixer 5.000.000
  - Verhindert "ewig überleben ohne fressen" als stabile Strategie
  - Umsetzung: `config.max_turns` nach jedem gefressenen Essen dynamisch erhöhen, im Agent

- [ ] **Score-Funktion vereinfachen** — `points_against_food` entfernen, nur `food_eaten` + Überlebensboni
  - Aktuell: Schlange wird für jeden Schritt weg vom Essen bestraft — auch wenn sie einer Bombe ausweicht
  - Führt zu Lernwiderspruch; sparsamere Rewards sind stabiler, aber anfangs schwerer zu lernen
  - Das Preset `in_game_score` geht bereits in diese Richtung, ist aber noch nicht ausgewertet

- [ ] **Speziation (echtes NEAT)** — Netzwerke nach Topologie in Species gruppieren, innerhalb Species selektieren
  - Größter fehlender Baustein: ohne Speziation werden strukturelle Mutationen bestraft,
    bevor sie sich beweisen können (Temp-Elite ist nur ein Behelf dafür)
  - Hoher Implementierungsaufwand; würde Fitness-Sharing und Species-Tracking erfordern

---

## Experimente

- [ ] Verschiedene Input-Profile testen und CSV-Reports vergleichen
  - Baseline: `bomb_aware` (12) → `timer` (15) → `full` (19) → `raw` (108)
- [ ] `LEVEL_ROTATION = True` vs. `False` vergleichen (Generalisierung vs. Spezialisierung)
- [ ] Scoring-Presets gegeneinander laufen lassen, besonders `in_game_score` vs. `balanced`

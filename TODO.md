# Bomberman Snake — TODO

## Bugs

- [x] **Body-Kollision in `can_move(LEFT/RIGHT)`** (`sensors.py`)
  - `abs_dir` wird jetzt korrekt für LEFT/RIGHT berechnet, bevor `_step()` aufgerufen wird

---

## Neue Inputs

Alle 14 Inputs sind in `sensors.py` + `SENSOR_FUNCS` + `profiles.py` implementiert.

- [x] **Food-Dringlichkeit** — `game.food_timer / FOOD_STEPS` (0.0–1.0)
- [x] **Bomb-Dringlichkeit** — `game.bomb_timer / BOMB_STEPS` (0.0–1.0)
- [x] **Explosion aktiv** — `1 if game.explosion else 0`
- [x] **Distanz zum Essen** — `manhattan(head, food_pos) / (2 * FIELDSIZE)` (0.0–1.0)
- [x] **Schlangenlänge** — `len(game.snake) / (FIELDSIZE * FIELDSIZE)` (0.0–1.0)

Profile: `minimal` (3), `basic` (6), `bomb_aware` (9), `timer` (12), `full` (14)

---

## Score-Tuning (`constants.py:ScoreConfig`)

- [x] `points_against_food = -0.5` — Bomben ausweichen wird nicht stark bestraft
- [x] `points_survived_tick = 0.01` — kleiner Überlebens-Bonus pro Schritt
- [x] `points_bomb_exploded = -5` — Schlange lernt aktiv, Bomben zu verhindern

---

## eternity.py

- [x] Zwischenstand speichern (Checkpoint nach Bootstrap-Erfolg und jeder Evo-Verbesserung)
- [x] Temp-Elite-System (Top 5 Kandidaten, müssen `TEMP_PROMOTE_AFTER` Runden > schlechtester Perm-Elite bleiben)
- [x] GPU-Frage → kein Vorteil (NEAT variable Topologie, sequenzieller Game-Loop; CPU-Multiprocessing ist richtig)
- [x] Seed-Offspring-Fix (neue Gehirne via `getOffspring(seeds)` + `mutatePopulation` statt Random)
- [x] Stabile Elite-Pool (nur via TempElite-Promotion austauschbar, außer in Generation 0)

---

## Experimente

- [ ] Verschiedene Input-Kombinationen testen und CSV-Reports vergleichen
  - Baseline: `bomb_aware` (9 Inputs)
  - +Timer-Inputs: `timer` (12 Inputs)
  - Alle Inputs: `full` (14 Inputs)

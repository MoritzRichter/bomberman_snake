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
- [x] Multi-Game-Evaluation (`EVAL_GAMES = 3`): Score über 3 Spiele mitteln → weniger Zufalls-Rauschen bei Selektion
- [x] Level-Rotation (`LEVEL_ROTATION = False`): opt-in, spielt ein Spiel pro Level → robustere Agenten

---

## Verbesserungen (offen)

- [ ] **Dynamisches Turn-Limit** — `max_turns = Basis + food_eaten × Bonus` statt fixer 5.000.000
  - Verhindert "ewig überleben ohne fressen" als stabile Strategie
  - Umsetzung: `config.max_turns` nach jedem gefressenen Essen dynamisch erhöhen innerhalb des Agents

- [ ] **Tournament-Selection** — K zufällige Gehirne ziehen, besten als Elternteil wählen
  - Robuster gegen Score-Ausreißer, mehr Populationsdiversität
  - Parameter: K=5–7; in `selection.py` als neue Strategie `"tournament"` ergänzen

- [ ] **Score-Funktion vereinfachen** — `points_against_food` entfernen, nur `food_eaten` + Überlebensboni
  - Aktuell: Schlange wird für jeden Schritt weg vom Essen bestraft — auch wenn sie Bombe ausweicht
  - Führt zu Lernwiderspruch; sparsamere Rewards sind stabiler, aber anfangs schwerer zu lernen

- [ ] **Speziation (echtes NEAT)** — Netzwerke nach Topologie in Species gruppieren, innerhalb Species selektieren
  - Größter fehlender Baustein: ohne Speziation werden strukturelle Mutationen sofort bestraft bevor sie sich beweisen können
  - Hoher Implementierungsaufwand; würde Fitness-Sharing und Species-Tracking erfordern

---

## Experimente

- [ ] Verschiedene Input-Kombinationen testen und CSV-Reports vergleichen
  - Baseline: `bomb_aware` (12 Inputs)
  - +Timer-Inputs: `timer` (15 Inputs)
  - Alle Inputs: `full` (17 Inputs)
- [ ] `LEVEL_ROTATION = True` vs. `False` vergleichen (Generalisierung vs. Spezialisierung)

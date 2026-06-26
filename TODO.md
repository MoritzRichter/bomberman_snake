# Bomberman Snake — TODO

## Bugs

- [ ] **Body-Kollision in `can_move(LEFT/RIGHT)`** (`sensors.py:56-59`)
  - `_is_collision` berechnet immer den Forward-Schritt statt die angefragte Richtung
  - Snake dreht sich und bewegt sich im selben Schritt → LEFT/RIGHT müssen die jeweilige Zelle prüfen
  - Praktisch kleiner Effekt, aber logisch falsch

---

## Neue Inputs

Aktuell: 9 Inputs (`can_move` ×3, `is_food` ×3, `is_bomb` ×3)  
Alle neuen Inputs müssen in `sensors.py:get_inputs()` ergänzt und `INPUT_SIZE` in `train.py` angepasst werden.

- [ ] **Food-Dringlichkeit** — `game.food_timer / FOOD_STEPS` (0.0–1.0)
  - Schlange lernt, Essen bei Timer≈1.0 zu meiden und wegzulaufen
- [ ] **Bomb-Dringlichkeit** — `game.bomb_timer / BOMB_STEPS` (0.0–1.0)
  - Schlange lernt, wann eine Explosion unmittelbar bevorsteht
- [ ] **Explosion aktiv** — `1 if game.explosion else 0`
  - Unterschied zwischen "ruhig" und "EXPLODED-Zellen auf Grid"
- [ ] **Distanz zum Essen** — `manhattan(head, food_pos) / (2 * FIELDSIZE)` (0.0–1.0)
  - Statt nur Richtung weiß die Schlange wie weit das Essen entfernt ist
- [ ] **Schlangenlänge** — `len(game.snake) / (FIELDSIZE * FIELDSIZE)` (0.0–1.0)
  - Einschätzung des Selbstkollisionsrisikos bei langer Schlange

---

## Score-Tuning (`constants.py:ScoreConfig`)

Aktuell:
```
points_towards_food = 1
points_against_food = -1.5
points_ate_food     = 2
```

- [ ] `points_against_food` senken (z.B. `-0.5`) — Schlange soll Bomben ausweichen dürfen ohne stark bestraft zu werden
- [ ] Überlebens-Bonus hinzufügen (`+0.01` pro Tick) — verhindert dass schnelles Sterben eine sinnvolle Strategie wird
- [ ] Bomben-Explosions-Malus hinzufügen (`-5`) — Schlange lernt aktiv, Bomben zu verhindern

---

## Experimente

- [ ] Verschiedene Input-Kombinationen testen und CSV-Reports vergleichen
  - Baseline: aktuelle 9 Inputs
  - +Timer-Inputs: food_timer + bomb_timer
  - +Distanz: food distance
  - Alle neuen Inputs kombiniert

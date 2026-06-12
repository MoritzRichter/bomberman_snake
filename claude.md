# Projekt Bomberman Snake

Das zugrunde liegende Spiel in `cg_base/` ist ein C/OpenGL-Programm — eine Mischung aus Bomberman und Snake, bei der das Essen nach kurzer Zeit in einer +-Form explodiert. Es läuft derzeit nur auf Linux.

## Ziel

Das Spiel soll als Python-Umgebung für ein Machine-Learning-Projekt neu implementiert werden, auf dem ein genetischer Algorithmus trainiert wird.

## Architektur: Zwei getrennte Schichten

Rendering und Spiellogik müssen strikt getrennt sein, damit der genetische Algorithmus tausende Simulationen ohne Rendering-Overhead durchlaufen kann.

### 1. `game_env.py` — Spiellogik (headless)

- Portierung der gesamten Logik aus `cg_base/src/logic.c` und `cg_base/src/level.c`
- Kein Pygame, kein Fenster, kein Rendering
- Implementiert das **OpenAI Gymnasium**-Interface:
  - `reset()` → startet eine neue Episode, gibt den initialen Zustand zurück
  - `step(action)` → führt einen Schritt aus, gibt `(observation, reward, terminated, truncated, info)` zurück
  - `observation_space` und `action_space` als Gymnasium-Spaces definiert
- Aktionen: 4 Richtungen (links, rechts, oben, unten)
- Observation: Grid-Zustand (10×10) + Schlangenpositionen + Bombe/Explosion-Status

### 2. `renderer.py` — Visualisierung (optional, Pygame)

- Nur zum Debuggen und Visualisieren, wird nicht beim Training verwendet
- Zeichnet das Spielfeld ähnlich wie die 2D-Ansicht in `cg_base/src/scene2D.c`
- Wird über `env.render(mode="human")` aktiviert

## Spiellogik (aus C-Quellcode)

- **Grid**: 10×10 Felder (`FIELDSIZE = 10`), Typen: `FT_FREE`, `FT_WALL`, `FT_EXPLODED`
- **Schlange**: Linked-List-artige Bewegung, wächst beim Fressen, kann nicht rückwärts laufen, Wrap-around an Rändern
- **Essen**: Erscheint zufällig auf freien Feldern, verschwindet nach `FOOD_COUNTER` Sekunden
- **Bombe**: Entsteht, wenn Essen nicht rechtzeitig gefressen wird, explodiert nach `BOMB_COUNTER` Sekunden
- **Explosion**: Breitet sich in +-Form mit Reichweite `EXPLOSION_REACH = 3` aus, blockiert durch Wände, verschwindet nach `EXPLOSION_COUNTER` Sekunden; trifft die Schlange → Schwanz wird abgeschnitten
- **Game Over**: Schlange läuft in eine Wand oder in sich selbst

## Level

Drei Level aus `cg_base/src/level.c` werden übernommen:
- Level 1: Einfaches Spielfeld mit Außenwänden
- Level 2: Offene Ecken, Wände in der Mitte
- Level 3: Gitter-artiges Layout

## Zeitkonstanten (aus `logic.c`)

```
STEPS_COUNTER      = 1.0   # Schritte pro Sekunde
FOOD_COUNTER       = 6.0   # Sekunden bis Bombe
BOMB_COUNTER       = 3.0   # Sekunden bis Explosion
EXPLOSION_COUNTER  = 2.0   # Sekunden bis Explosion verschwindet
EXPLOSION_REACH    = 3     # Felder Reichweite der Explosion
```

Im headless-Modus wird Zeit als diskrete Schritt-Zählung modelliert (kein `time.sleep`).
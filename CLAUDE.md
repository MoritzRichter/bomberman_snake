# Projekt Bomberman Snake

Python-Neuimplementierung von **Bomberman Snake** (Snake, bei dem das Essen nach kurzer Zeit
in einer `+`-Form explodiert) plus ein genetischer Algorithmus, der neuronale Netze mit
variabler Topologie darauf trainiert.

Vollständige Doku: [README.md](README.md). Diese Datei enthält nur, was beim Arbeiten am
Code leicht übersehen wird.

## Architektur

Rendering und Spiellogik sind strikt getrennt, damit der genetische Algorithmus tausende
Simulationen ohne Rendering-Overhead durchlaufen kann.

- `game/logic.py` — `GameLogic`, headless, **kein pygame**. Portiert aus `cg_base/src/logic.c`.
- `game/renderer.py` — pygame, nur zum Debuggen und Zuschauen.
- `evolution/` — GA + Netze.

Es gibt **kein** Gymnasium-Interface. `GameLogic.step(action)` gibt `(ate_food, died)` zurück,
nicht das Gym-5-Tupel. Aktionen sind absolute Richtungen (`Direction`, 0–3); die trainierten
Netze steuern dagegen relativ (links / rechts / geradeaus) und rechnen selbst um.

## Fallstricke

**Pickle-Kompatibilität — wichtigste Regel.** Die Skripte legen `evolution/` per
`sys.path.insert` auf den Pfad und importieren flach (`from network import ...`). Das ist
historisch gewachsen und **muss so bleiben**: die gespeicherten `.pkl`-Dateien referenzieren
die Klassen als `network.Network` / `network.Node`. Ein Umbau von `evolution/` zu einem
echten Package mit relativen Imports macht sämtliche gespeicherten Netze unladbar.

**Zwei `constants.py`.** `game/constants.py` (Spielregeln) und `evolution/constants.py`
(GA-Konfiguration). Wegen der flachen Imports meint `from constants import config` immer
das aus `evolution/`.

**`evolution/fast_eval.py` dupliziert Konstanten.** Numba kann nicht auf Python-Objekte
zugreifen, deshalb spiegelt das Modul `FIELDSIZE`, `FOOD_STEPS`, `BOMB_STEPS`,
`EXPLOSION_STEPS`, `EXPLOSION_REACH` und die Sensor-Reihenfolge als eigene Konstanten.
Ändert sich eine Spielkonstante oder die Reihenfolge in `SENSOR_FUNCS`, muss `fast_eval.py`
mitgezogen werden — sonst weichen JIT- und Python-Pfad still voneinander ab.

**Koordinaten.** `grid[y][x]`, `y=0` ist die **unterste** Reihe (OpenGL-Konvention aus dem
C-Original). `UP` bedeutet `y+1`. An allen Rändern gilt Wrap-around (`% FIELDSIZE`).

## Konstanten (aus `logic.c`, headless als diskrete Schrittzahlen)

```
FIELDSIZE       = 10    FOOD_STEPS      = 6    # Schritte bis Bombe
EXPLOSION_REACH = 3     BOMB_STEPS      = 3    # Schritte bis Explosion
START_X/Y       = 6     EXPLOSION_STEPS = 2    # Schritte bis Explosion verschwindet
```

## Tests

```bash
python tests/test_sensors.py
python tests/test_mutations.py
```

## `cg_base/`

Das ursprüngliche C/OpenGL-Programm (nur Linux). Liegt lokal als Referenz für die Portierung,
ist per `.gitignore` bewusst nicht versioniert.

# Bomberman Snake – Implementierungsanleitung

## Projektübersicht

Das Ziel ist eine Python-Neuimplementierung des C/OpenGL-Spiels als Gymnasium-Environment,
auf dem ein genetischer Algorithmus trainiert wird. Die Architektur ist strikt in zwei
Schichten getrennt: headless Spiellogik (für Training) und optionaler Pygame-Renderer
(für Visualisierung).

---

## Dateistruktur (Zielzustand)

```
bomberman_snake/
├── game/
│   ├── __init__.py          # leer, nur Package-Marker
│   ├── constants.py         # ✅ bereits fertig
│   ├── levels.py            # Schritt 1 – Level-Layouts
│   ├── logic.py             # Schritt 2 – Spiellogik (headless)
│   ├── env.py               # Schritt 3 – Gymnasium-Wrapper
│   └── renderer.py          # Schritt 5 – Pygame-Renderer
├── ml/
│   ├── __init__.py          # leer
│   └── genetic.py           # Schritt 4 – Genetischer Algorithmus + Netzwerk
├── main.py                  # Schritt 6 – Manuelles Spielen
├── train.py                 # Schritt 7 – Training
├── requirements.txt         # ✅ bereits fertig
└── claude.md / CLAUDE.md    # ✅ Projektdokumentation
```

---

## Koordinatensystem (wichtig für alle Schritte)

Das C-Original speichert das Grid als `grid[y][x]`, wobei `y=0` die **unterste** Reihe
ist (OpenGL: y-Achse zeigt nach oben). Python übernimmt dieselbe Konvention.

| Richtung | x-Delta | y-Delta | Wrap-around |
|----------|---------|---------|-------------|
| LEFT     | −1      | 0       | x < 0 → x + 10 |
| RIGHT    | +1      | 0       | x ≥ 10 → x − 10 |
| UP       | 0       | +1      | y ≥ 10 → y − 10 |
| DOWN     | 0       | −1      | y < 0 → y + 10 |

Für den Pygame-Renderer muss die y-Achse gespiegelt werden:
`pygame_row = FIELDSIZE − 1 − game_y`

---

## Timing-Modell (headless, diskret)

Jeder `step()`-Aufruf = eine Schlangenbewegung.

| Konstante         | Wert (Schritte) | Bedeutung                          |
|-------------------|-----------------|------------------------------------|
| `FOOD_STEPS`      | 6               | Schritte bis Essen → Bombe         |
| `BOMB_STEPS`      | 3               | Schritte bis Bombe → Explosion     |
| `EXPLOSION_STEPS` | 2               | Schritte bis Explosion verschwindet|
| `EXPLOSION_REACH` | 3               | Felder Reichweite (+-Form)         |

### Reihenfolge pro Step (muss exakt so sein!)

```
1. Schlange bewegen   → neue Kopfposition berechnen, Kollision prüfen
2. Schwanz abschneiden → Segmente auf EXPLODED-Feldern entfernen
3. Explosion ticken   → Timer erhöhen; wenn abgelaufen: Grid bereinigen
4. Essen ticken       → Timer erhöhen; wenn abgelaufen: Bombe spawnen
5. Bombe ticken       → Timer erhöhen; wenn abgelaufen: Explosion starten + Grid markieren
```

Diese Reihenfolge stellt sicher:
- Eine neue Explosion blockiert erst **nächsten** Schritt (1-Schritt Verzögerung, wie im C-Code)
- Essen das in diesem Schritt gefressen wird, erzeugt **keine** Bombe
- Explosion dauert exakt 2 volle Schritte

---

## Schritt 1 – `game/levels.py`

**Abhängigkeiten:** `constants.py`

**Aufgabe:** Die drei Level aus `cg_base/src/level.c` als Python-Listen portieren.

### Was zu implementieren ist

```python
from .constants import FieldType

W = FieldType.WALL
_ = FieldType.FREE

# Jedes Level: Liste von 10 Reihen, Reihe 0 = unterste Reihe (y=0)
LEVEL_1 = [
    [W, W, W, W, W, W, W, W, W, W],  # y=0 (unten)
    [W, _, _, _, _, _, _, _, _, W],
    ...
    [W, W, W, W, W, W, W, W, W, W],  # y=9 (oben)
]

LEVEL_2 = [...]  # Offene Ecken, Wände in der Mitte
LEVEL_3 = [...]  # Gitter-Layout

LEVELS = {1: LEVEL_1, 2: LEVEL_2, 3: LEVEL_3}
```

### Verifikation

Nach Implementierung prüfen:
- `LEVELS[1][0]` → alle Wände (unterste Reihe)
- `LEVELS[1][6][6]` → `FieldType.FREE` (Startposition der Schlange)
- `LEVELS[1][6][4]` → `FieldType.FREE` (Schwanzposition der Schlange)
- Alle drei Level: Positionen (4,6), (5,6), (6,6) müssen FREE sein

---

## Schritt 2 – `game/logic.py`

**Abhängigkeiten:** `constants.py`, `levels.py`

**Aufgabe:** Vollständige headless Spiellogik als Klasse `GameLogic`.

### Klassenstruktur

```python
class GameLogic:
    # Zustand
    grid: list[list[FieldType]]   # 10×10, grid[y][x]
    snake: list[tuple[int,int]]   # [(hx,hy), (b1x,b1y), ..., (tx,ty)]
    direction: Direction
    food_pos: tuple[int,int]
    bomb: bool
    bomb_pos: tuple[int,int] | None
    bomb_timer: int
    explosion: bool
    explosion_pos: tuple[int,int] | None
    explosion_timer: int
    food_timer: int
    game_over: bool
    score: int
```

### Methoden (in dieser Reihenfolge implementieren)

#### `reset()` → None
- Grid aus Level-Layout kopieren (deep copy!)
- Schlange initialisieren: 3 Teile bei (6,6), (5,6), (4,6), Richtung RIGHT
- Alle Timer und Flags zurücksetzen
- Zufälliges Essen platzieren

#### `_place_food()` → None
- Zufällige freie Position finden (kein WALL, kein EXPLODED, nicht auf Schlange)
- Per `random.randrange(FIELDSIZE)` in einer Schleife suchen

#### `_move_snake(action)` → `(ate_food: bool, died: bool)`
- Neue Richtung setzen (keine Umkehrung erlaubt, prüfen mit `OPPOSITE`-Dict)
- Neue Kopfposition: `(hx + dx) % FIELDSIZE`, `(hy + dy) % FIELDSIZE`
- **Game Over wenn:** `grid[ny][nx] != FieldType.FREE`
- **Game Over wenn:** `(nx, ny) in self.snake[1:]` (Selbstkollision, Kopf nicht zählen)
- Essen aufnehmen: `self.snake = [(nx, ny)] + self.snake` (kein Schwanz entfernen)
  - `food_timer = 0`, `score += 1`, neue Food-Position berechnen
- Normale Bewegung: `self.snake = [(nx, ny)] + self.snake[:-1]` (Schwanz entfernen)

#### `_cut_tail_at_explosion()` → None
- Erste EXPLODED-Position im Snake-Array finden
- `i == 0` (Kopf): `game_over = True`, Snake leeren
- `i > 0` (Körper): `self.snake = self.snake[:i]`

#### `_cross_field(cx, cy, field_type)` → None
- Setzt Mittelzelle: `grid[cy][cx] = field_type`
- 4 Richtungen (rechts, links, oben, unten), jeweils bis `EXPLOSION_REACH`:
  - Mit Wrap-around: `nx = (cx + i * dx) % FIELDSIZE`
  - Stopp bei WALL (break), sonst setzen

```python
# Beispiel für eine Richtung:
for i in range(1, EXPLOSION_REACH + 1):
    nx = (cx + i) % FIELDSIZE
    if self.grid[cy][nx] == FieldType.WALL:
        break
    self.grid[cy][nx] = field_type
```

#### `_apply_explosion()` → None
- `_cross_field(explosion_pos.x, explosion_pos.y, FieldType.EXPLODED)`

#### `_clear_explosion()` → None
- `_cross_field(explosion_pos.x, explosion_pos.y, FieldType.FREE)`

#### `_tick_explosion()` → None
- Nur wenn `self.explosion`:
  - `explosion_timer += 1`
  - Wenn `>= EXPLOSION_STEPS`: `explosion = False`, `_clear_explosion()`

#### `_tick_food()` → None
- `food_timer += 1`
- Wenn `>= FOOD_STEPS`: Bombe spawnen
  - `bomb = True`, `bomb_pos = food_pos`, `bomb_timer = 0`
  - Neues Essen platzieren (`_place_food()`)
  - `food_timer = 0`

#### `_tick_bomb()` → None
- Nur wenn `self.bomb`:
  - `bomb_timer += 1`
  - Wenn `>= BOMB_STEPS`:
    - `bomb = False`, `bomb_timer = 0`
    - `explosion = True`, `explosion_pos = bomb_pos`, `explosion_timer = 0`
    - `_apply_explosion()`

#### `step(action)` → `(ate_food: bool, died: bool)`
```python
def step(self, action):
    if self.game_over:
        return False, True
    ate_food, died = self._move_snake(action)
    if not died:
        self._cut_tail_at_explosion()
        self._tick_explosion()        # erst Explosion ticken ...
        if not ate_food:
            self._tick_food()         # ... dann Essen ...
        self._tick_bomb()             # ... dann Bombe (kann neue Explosion starten)
    return ate_food, died or self.game_over
```

### Verifikation

```python
# Schnelltest nach Implementierung
from game.logic import GameLogic
from game.constants import Direction

g = GameLogic(level=1)
assert len(g.snake) == 3
assert g.snake[0] == (6, 6)
assert g.snake[2] == (4, 6)
assert g.food_pos is not None

# Schritt machen
ate, died = g.step(Direction.RIGHT)
assert g.snake[0] == (7, 6)   # Kopf bewegt sich nach rechts
assert len(g.snake) == 3       # Länge gleich (kein Essen)
assert not died

# 6 Schritte ohne Essen → Bombe spawnt
g2 = GameLogic(level=1)
g2.food_pos = (9, 9)  # weit weg setzen
for _ in range(6):
    g2.step(Direction.RIGHT)
assert g2.bomb == True
```

---

## Schritt 3 – `game/env.py`

**Abhängigkeiten:** `logic.py`, `constants.py`

**Aufgabe:** `GameLogic` als Gymnasium-Environment wrappen.

### Klasse `BombermanSnakeEnv(gymnasium.Env)`

#### Observation Space
```python
spaces.Box(low=0, high=6, shape=(FIELDSIZE, FIELDSIZE), dtype=np.int8)
```
Zellwerte (aus `Cell`-Enum in `constants.py`):
- 0 = FREE, 1 = WALL, 2 = EXPLODED
- 3 = SNAKE_BODY, 4 = SNAKE_HEAD, 5 = FOOD, 6 = BOMB

#### Action Space
```python
spaces.Discrete(4)   # 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
```

#### `reset(seed, options)` → `(obs, info)`
- Neues `GameLogic`-Objekt erstellen
- Schritt-Zähler auf 0
- Observation und Info zurückgeben

#### `step(action)` → `(obs, reward, terminated, truncated, info)`
- `self._game.step(action)` aufrufen
- Reward-Schema:
  - `+1.0` für Essen gefressen
  - `-1.0` für Game Over (Kollision)
  - `0.0` sonst
- `terminated = died`
- `truncated = step_count >= max_steps`

#### `_get_obs()` → `np.ndarray`
```python
grid = np.zeros((FIELDSIZE, FIELDSIZE), dtype=np.int8)
# 1. Terrain aus self._game.grid kopieren (FREE/WALL/EXPLODED)
# 2. Food einzeichnen
# 3. Bomb einzeichnen (falls aktiv)
# 4. Snake-Body einzeichnen
# 5. Snake-Head einzeichnen (überschreibt Body falls Überlapp)
return grid
```

#### `render()` → None / np.ndarray
- Nur wenn `render_mode` gesetzt: Renderer importieren und aufrufen
- Lazy import um Pygame-Abhängigkeit zu vermeiden wenn nicht gerendert wird

### Verifikation

```python
import gymnasium as gym
import numpy as np
from game.env import BombermanSnakeEnv

env = BombermanSnakeEnv(level=1)
obs, info = env.reset()
assert obs.shape == (10, 10)
assert obs.dtype == np.int8

obs, reward, terminated, truncated, info = env.step(1)  # RIGHT
assert obs.shape == (10, 10)
assert not terminated

# Gymnasium-Kompatibilität prüfen
from gymnasium.utils.env_checker import check_env
check_env(env, warn=True)
```

---

## Schritt 4 – `ml/genetic.py`

**Abhängigkeiten:** `game/env.py`, numpy

**Aufgabe:** Genetischen Algorithmus + neuronales Netzwerk für Entscheidungen.

### Netzwerkarchitektur (`NeuralNetwork`)

```
Input:  100  (10×10 Grid, flatten, normalisiert 0–1)
Hidden:  64  (tanh-Aktivierung)
Output:   4  (Linear, argmax für Aktion)

Parameter: 100×64 + 64 + 64×4 + 4 = 6.724
Genome:     flat numpy array [W1|b1|W2|b2] der Länge 6.724
```

#### Methoden

```python
class NeuralNetwork:
    GENOME_SIZE = 6724

    def __init__(self, genome: np.ndarray): ...

    @classmethod
    def random(cls) -> 'NeuralNetwork':
        # genome = np.random.randn(cls.GENOME_SIZE) * 0.1
        ...

    def predict(self, obs: np.ndarray) -> int:
        # obs flattened & normalisiert → Forward pass → argmax
        ...
```

### Genetischer Algorithmus (`GeneticAlgorithm`)

```python
class GeneticAlgorithm:
    def __init__(self, pop_size=100, level=1, max_steps=500,
                 mutation_rate=0.1, mutation_sigma=0.05,
                 tournament_k=5, elite_count=2): ...

    def evaluate(self, genome: np.ndarray) -> float:
        # Eine Episode mit BombermanSnakeEnv durchlaufen
        # Fitness = Summe der Rewards (entspricht Score)
        ...

    def tournament_select(self) -> np.ndarray:
        # k zufällige Individuen, bestes Genome zurückgeben
        ...

    def crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Uniform crossover: jedes Gen mit 50% von a oder b
        mask = np.random.rand(len(a)) < 0.5
        return np.where(mask, a, b)

    def mutate(self, genome: np.ndarray) -> np.ndarray:
        # Gaußsches Rauschen mit Wahrscheinlichkeit mutation_rate
        mask = np.random.rand(len(genome)) < self.mutation_rate
        noise = np.random.randn(len(genome)) * self.mutation_sigma
        return genome + mask * noise

    def evolve_one_generation(self) -> dict:
        # 1. Fitness aller Individuen berechnen
        # 2. Elite direkt übernehmen
        # 3. Rest: select → crossover → mutate
        # Gibt Statistiken zurück: {best, mean, worst}
        ...

    def run(self, generations: int, callback=None): ...
    def save(self, path: str): ...
    def load(self, path: str): ...
```

### Verifikation

```python
from ml.genetic import NeuralNetwork, GeneticAlgorithm
import numpy as np

nn = NeuralNetwork.random()
obs = np.zeros((10, 10), dtype=np.int8)
action = nn.predict(obs)
assert 0 <= action <= 3

ga = GeneticAlgorithm(pop_size=10, max_steps=100)
stats = ga.evolve_one_generation()
assert 'best' in stats and 'mean' in stats
```

---

## Schritt 5 – `game/renderer.py`

**Abhängigkeiten:** `logic.py`, `constants.py`, pygame

**Aufgabe:** Optionaler Pygame-Renderer zur Visualisierung.

### Klasse `Renderer`

#### Konfiguration

```python
CELL_SIZE = 60        # Pixel pro Zelle
WINDOW_SIZE = FIELDSIZE * CELL_SIZE   # 600×600 Pixel

COLORS = {
    Cell.FREE:       (80,  80,  80),
    Cell.WALL:       (60,  60,  60),   # dunkelgrau, optional Ziegel
    Cell.EXPLODED:   (200, 80,  20),   # orange-rot
    Cell.SNAKE_HEAD: (0,   200, 0),    # hell-grün
    Cell.SNAKE_BODY: (0,   160, 0),    # dunkel-grün
    Cell.FOOD:       (220, 50,  50),   # rot
    Cell.BOMB:       (40,  40,  40),   # fast schwarz
}
```

#### Methoden

```python
class Renderer:
    def __init__(self, cell_size: int = 60): ...
    def draw(self, game: GameLogic): ...
    def get_rgb_array(self, game: GameLogic) -> np.ndarray: ...
    def close(self): ...
```

#### `draw(game)` – Implementierungsreihenfolge

```
1. Observation-Grid aus GameLogic bauen (wie in env._get_obs())
2. Für jede Zelle (x, y):
   - pygame_row = FIELDSIZE - 1 - y   (y-Achse spiegeln!)
   - rect = (x * CELL_SIZE, pygame_row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
   - pygame.draw.rect(surface, COLORS[cell], rect)
   - Optional: kleines Padding (rect.inflate(-4, -4)) für Grid-Optik
3. pygame.display.flip()
4. Clock auf ~10 FPS begrenzen (beim Training nicht nötig)
```

### Verifikation

```python
# Manuell starten:
from game.logic import GameLogic
from game.renderer import Renderer

g = GameLogic(level=1)
r = Renderer()
r.draw(g)  # Fenster sollte erscheinen
```

---

## Schritt 6 – `main.py`

**Abhängigkeiten:** `game/env.py`, `game/renderer.py`, pygame

**Aufgabe:** Manuelles Spielen mit Pfeiltasten.

### Programmfluss

```python
def main():
    env = BombermanSnakeEnv(level=1, render_mode="human")
    obs, _ = env.reset()

    running = True
    action = Direction.RIGHT   # Startrichtung

    while running:
        # 1. Pygame Events verarbeiten
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:  action = Direction.LEFT
                if event.key == pygame.K_RIGHT: action = Direction.RIGHT
                if event.key == pygame.K_UP:    action = Direction.UP
                if event.key == pygame.K_DOWN:  action = Direction.DOWN
                if event.key == pygame.K_r:     obs, _ = env.reset()
                if event.key == pygame.K_ESCAPE: running = False

        # 2. Schritt ausführen
        obs, reward, terminated, truncated, info = env.step(action)

        # 3. Score anzeigen (Pygame-Title oder Konsole)
        pygame.display.set_caption(f"Bomberman Snake – Score: {info['score']}")

        # 4. Bei Game Over kurz warten, dann neu starten
        if terminated or truncated:
            pygame.time.wait(1000)
            obs, _ = env.reset()

        # 5. FPS begrenzen (z.B. 5 FPS für spielbares Tempo)
        clock.tick(5)

    env.close()
```

---

## Schritt 7 – `train.py`

**Abhängigkeiten:** `ml/genetic.py`

**Aufgabe:** Trainingsloop mit Fortschrittsanzeige und Modell-Speicherung.

### Programmfluss

```python
def main():
    ga = GeneticAlgorithm(
        pop_size=100,
        level=1,
        max_steps=500,
        mutation_rate=0.1,
        mutation_sigma=0.05,
        tournament_k=5,
        elite_count=2,
    )

    best_ever = -float('inf')

    for gen in range(500):
        stats = ga.evolve_one_generation()

        print(f"Gen {gen:4d} | Best: {stats['best']:.1f} | "
              f"Mean: {stats['mean']:.2f} | Worst: {stats['worst']:.1f}")

        # Bestes Modell speichern wenn verbessert
        if stats['best'] > best_ever:
            best_ever = stats['best']
            ga.save('best_genome.npy')
            print(f"         → Neues Bestmodell gespeichert (Score: {best_ever:.1f})")

    print("Training abgeschlossen.")
```

---

## Empfohlene Testreihenfolge

```
Schritt 1:  python -c "from game.levels import LEVELS; print(LEVELS[1][6][6])"
            # Erwartet: FieldType.FREE

Schritt 2:  python -c "
            from game.logic import GameLogic
            from game.constants import Direction
            g = GameLogic(); g.step(Direction.RIGHT)
            print(g.snake[0])   # (7, 6)
            "

Schritt 3:  python -c "
            from game.env import BombermanSnakeEnv
            from gymnasium.utils.env_checker import check_env
            env = BombermanSnakeEnv()
            check_env(env)
            print('Gymnasium-Check bestanden')
            "

Schritt 4:  python -c "
            from ml.genetic import GeneticAlgorithm
            ga = GeneticAlgorithm(pop_size=5, max_steps=50)
            stats = ga.evolve_one_generation()
            print(stats)
            "

Schritt 5:  python main.py        # Fenster öffnet sich, Pfeiltasten steuern
Schritt 6:  python train.py       # Trainingsoutput in Konsole
```

---

## Häufige Fallstricke

| Problem | Ursache | Lösung |
|---------|---------|--------|
| Snake verschwindet sofort | Starposition liegt auf Wand | Startpos (6,6) im Level prüfen |
| Explosion löscht Wände | `_cross_field` setzt WALL auf FREE | Nur nicht-WALL Felder zurücksetzen |
| Food spawnt auf Snake | `_place_food` prüft Snake nicht | `(x,y) not in self.snake` hinzufügen |
| Y-Achse gespiegelt im Renderer | Pygame y≠OpenGL y | `pygame_row = FIELDSIZE-1-y` verwenden |
| Training divergiert sofort | Netzgewichte zu groß | Gewichte mit 0.1 skalieren |
| `check_env` schlägt fehl | Falscher Observation dtype | `dtype=np.int8` in Box verwenden |

---

## Abhängigkeitsdiagramm

```
constants.py
    │
    ├── levels.py
    │       │
    │       └── logic.py
    │               │
    │               ├── env.py ──── renderer.py
    │               │       │
    │               │       └── ml/genetic.py
    │               │
    │               └── renderer.py
    │
    └── [alle anderen]

main.py   →  env.py + renderer.py
train.py  →  ml/genetic.py  →  env.py
```

Jede Datei kann erst implementiert werden, wenn alle Abhängigkeiten darüber fertig sind.

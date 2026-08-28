# Bomberman Snake — Neuroevolution

Eine Python-Neuimplementierung des Spiels **Bomberman Snake** (Snake mit explodierendem
Essen) samt einem **genetischen Algorithmus**, der neuronale Netze mit variabler Topologie
(NEAT-artig) darauf trainiert.

Uni-Gruppenarbeit von **Bryan Buck** (stud104897), **Moritz Richter** (stud104055)
und **Daven Vogel** (stud104819).

Das Original ist ein C/OpenGL-Programm (`cg_base/`, nur Linux). Diese Neuimplementierung
trennt **Spiellogik** und **Rendering** strikt, damit der genetische Algorithmus tausende
Simulationen ohne Rendering-Overhead durchlaufen kann.

---

## Inhalt

- [Schnellstart](#schnellstart)
- [Projektstruktur](#projektstruktur)
- [Die Programme](#die-programme)
- [Spielregeln](#spielregeln)
- [Der genetische Algorithmus](#der-genetische-algorithmus)
- [Trainings-Workflow](#trainings-workflow)
- [Ausgabedateien](#ausgabedateien)
- [Performance](#performance)
- [Tests](#tests)
- [Technische Hinweise](#technische-hinweise)

---

## Schnellstart

```bash
pip install -r requirements.txt

python main.py             # selbst spielen
python play.py             # trainiertem Netz beim Spielen zusehen
python train.py            # Training mit Live-Visualisierung (600 Generationen)
python eternity_deep.py    # Dauertraining ohne Rendering (empfohlen)
```

**Abhängigkeiten:** `numpy`, `pygame`, `matplotlib`; optional `numba` (JIT, ~50× schneller —
ohne numba fällt `eternity_deep.py` automatisch auf den reinen Python-Pfad zurück).

---

## Projektstruktur

```
├── game/                    Spiellogik + Rendering (die zwei getrennten Schichten)
│   ├── constants.py           Feldgrößen, Timer, Enums (FieldType, Direction, Cell)
│   ├── levels.py              Die drei Level-Layouts
│   ├── logic.py               GameLogic — headless, kein pygame
│   └── renderer.py            GridRenderer + Renderer (pygame, optional)
│
├── evolution/               Genetischer Algorithmus + neuronale Netze
│   ├── network.py             Network / Node / Connection + Forward-Pass
│   ├── evolution.py           10 Mutationsoperatoren + Populationsoperationen
│   ├── crossover.py           Rekombination zweier Elternnetze
│   ├── selection.py           5 Selektionsstrategien
│   ├── sensors.py             MoveHelper — Spielzustand → Netzwerk-Inputs
│   ├── profiles.py            Sensor-Profile (welche Inputs, wie stark gewichtet)
│   ├── agent.py               Ein Gehirn + ein Spiel + Fitness-Berechnung
│   ├── constants.py           Config-Objekt, Score-Presets, Aktivierungsfunktionen
│   └── fast_eval.py           Numba-JIT-Variante des kompletten Spielloops
│
├── main.py                  Selbst spielen
├── play.py                  Trainiertes Netz zuschauen
├── train.py                 Training mit Live-Visualisierung
├── eternity.py              Dauertraining (flache Netze, Multiprocessing)
├── eternity_deep.py         Dauertraining (tiefe Netze, JIT, adaptive Mutation)
├── tuner.py                 Hyperparameter-Suche
├── inspect_network.py       Netzwerk-Topologie visuell inspizieren
├── plot_csv.py              Training-Reports als Graphen plotten
│
├── models/                  Gespeicherte Netze
│   ├── veterans/              Einzelne beste Netze aus train.py
│   ├── seeds/                 Elite-Populationen (Warm-Start)
│   ├── showcase/              Handverlesene Highlights
│   └── best_network.pkl
│
├── runs/                    Trainings-Ausgaben
│   ├── eternity/              Pakete aus eternity.py
│   ├── eternity_deep/         Pakete aus eternity_deep.py
│   ├── reports/               training_report_*.csv aus train.py
│   └── tuner/                 tuner_*.csv
│
├── docs/
│   ├── TODO.md                Offene Punkte & Ideen
│   └── update.md              Chronologisches Änderungslog
│
└── tests/                   Unit-Tests (unittest)
```

---

## Die Programme

| Skript | Zweck | Rendering |
|---|---|---|
| `main.py` | Selbst spielen (Pfeiltasten) | ja |
| `play.py` | Gespeichertem Netz beim Spielen zusehen | ja |
| `train.py` | Training mit Live-Ansicht aller 50 Spiele | ja |
| `eternity.py` | Dauertraining, flache Netze, `multiprocessing.Pool` | nein¹ |
| `eternity_deep.py` | Dauertraining, tiefe Netze, Numba-JIT + `ThreadPool` | nein¹ |
| `tuner.py` | Grob-zu-fein-Suche über Hyperparameter | nein |
| `inspect_network.py` | Topologie & Gewichte eines Netzes anschauen | ja |
| `plot_csv.py` | Training-Report-CSV als matplotlib-Graphen | ja |

¹ optional per `VISUAL = True` im Skript-Kopf.

### `main.py` — selbst spielen

| Taste | Wirkung |
|---|---|
| Pfeiltasten | Richtung ändern |
| `1` / `2` / `3` | Level wechseln |
| `E` | Essen-Explosion an/aus |
| `X` | Explosionsform `+` / `X` |
| `R` | Neustart |
| `ESC` / `Q` | Beenden |

### `play.py` — Netz zuschauen

Auswahlbildschirm listet alle `veteran*.pkl` aus `models/veterans/`, `runs/eternity/` und
`runs/eternity_deep/` (neueste zuerst), mit Level-Wahl und Optionen für Essen-Explosion,
Explosionsform und Ansicht.

| Taste | Wirkung |
|---|---|
| `↑` / `↓` | Geschwindigkeit (0.5 – 1000 Ticks/s) |
| `R` | Neustart |
| `ESC` | zurück zur Auswahl |
| `Q` | Beenden |

### `inspect_network.py` — Topologie inspizieren

Mausrad = Zoom, Linksziehen = Pan, Klick auf Knoten/Kante = Details in der Seitenleiste,
`T` / `Shift+T` = Gewichts-Schwelle, `L` = Gewichtslabels, `R` = Kamera zurücksetzen.

Aufruf mit Pfad möglich: `python inspect_network.py models/veterans/veteran_1.pkl`

### `plot_csv.py` — Reports plotten

```bash
python plot_csv.py                         # Datei-Dialog
python plot_csv.py runs/reports/xyz.csv    # direkt
python plot_csv.py -s                      # vereinfachte Ansicht
```

---

## Spielregeln

Portiert aus `cg_base/src/logic.c` und `cg_base/src/level.c`.

- **Grid:** 10×10, Feldtypen `FREE` / `WALL` / `EXPLODED`. `grid[y][x]`, `y=0` ist die
  **unterste** Reihe (OpenGL-Konvention aus dem C-Original).
- **Schlange:** Start bei `(6, 6)`, Länge 3, Richtung rechts. Wächst beim Fressen, kann nicht
  rückwärts laufen, **Wrap-around** an den Rändern (`% FIELDSIZE`).
- **Essen:** erscheint auf einer zufälligen freien Zelle.
- **Bombe:** wird das Essen nicht rechtzeitig gefressen, wird daraus eine Bombe und neues
  Essen erscheint.
- **Explosion:** breitet sich in `+`-Form (optional `X`) mit Reichweite 3 aus. Wände blocken
  den Strahl, werden aber nicht zerstört.
- **Schwanz abschneiden:** trifft die Explosion ein Körpersegment, wird die Schlange ab dort
  gekappt. Trifft sie den Kopf → sofort Game Over.
- **Game Over:** Kollision mit Wand, Explosionsfeld oder dem eigenen Körper.

**Schrittreihenfolge pro Tick:** `move` → `cut_tail` → `tick_explosion` → `tick_food` → `tick_bomb`

**Zeitkonstanten** (`game/constants.py`) — im Headless-Modus als diskrete Schrittzahlen, nicht als Sekunden:

```
FIELDSIZE       = 10     FOOD_STEPS      = 6    # Schritte bis Bombe
EXPLOSION_REACH = 3      BOMB_STEPS      = 3    # Schritte bis Explosion
START_X/Y       = 6      EXPLOSION_STEPS = 2    # Schritte bis Explosion verschwindet
```

**Level** (`game/levels.py`):

| Level | Layout |
|---|---|
| 1 | Einfaches Spielfeld mit Außenwänden |
| 2 | Offene Ecken, Wandblock in der Mitte |
| 3 | Gitter-artig, viele Durchgänge |

---

## Der genetische Algorithmus

### Netzwerk

NEAT-artig: eine Liste von Knoten in Aktivierungsreihenfolge plus eine Liste gerichteter
Verbindungen. Es gibt keine festen Layer — die Topologie entsteht durch Mutation.

- Start: alle Inputs voll verbunden mit den Outputs (He-Initialisierung), keine Hidden-Knoten
- Aktivierungsfunktionen pro Knoten: `sigmoid`, `tanh`, `relu`, `identity`
- **2 Outputs:** `turn_left`, `turn_right`. Beide < 0.5 → geradeaus. Die Netze steuern also
  **relativ** (links/rechts/geradeaus), nicht absolut.

### Sensoren & Profile

`evolution/sensors.py` stellt 19 handgebaute Sensoren bereit; `evolution/profiles.py` legt
fest, welche davon aktiv sind und wie stark sie gewichtet werden (Gewicht `0.0` = deaktiviert).

| Profil | Inputs | Inhalt |
|---|---|---|
| `minimal` | 3 | nur Kollisionsvermeidung (`can_move_*`) |
| `basic` | 6 | + Essensrichtung |
| `bomb_aware` | 12 | + Bombenrichtung + Körpernähe |
| `timer` | 15 | + Dringlichkeits-Timer + Explosion aktiv |
| `full` | 19 | alles |
| `raw` | 108 | **kein** Handcrafting: 100 Grid-Zellen + Richtungs-One-Hot + Timer + Länge |

`raw` gibt dem Netz den kompletten Spielzustand roh — mehr Freiheit, deutlich langsamere
Konvergenz. `eternity.py` nutzt `raw`, `eternity_deep.py` nutzt `full`.

### Fitness

Ein Agent sammelt Punkte über fünf Komponenten (`evolution/agent.py`):

| Komponente | wann |
|---|---|
| `points_towards_food` | Zug in Richtung Essen |
| `points_against_food` | Zug weg vom Essen |
| `points_ate_food` | Essen gefressen |
| `points_survived_tick` | pro überlebtem Schritt |
| `points_bomb_exploded` | Strafe, wenn eine Bombe hochgeht |
| `points_per_length_tick` | Bonus × Schlangenlänge pro Schritt |

Fünf Presets in `evolution/constants.py`, umschaltbar per `apply_scoring_preset(name)`:

`balanced` (Standard) · `survival` (lange leben) · `food` (schnell fressen) ·
`length` (lange Schlange) · `in_game_score` (nur der echte Spielscore)

### Selektion

`power` (Standard, x^k-Verzerrung Richtung Spitze) · `tournament` · `roulette` ·
`top_n` · `random` (Kontrolle). Siehe `evolution/selection.py`.

### Mutation

Zehn Operatoren mit relativen Wahrscheinlichkeiten (`MUTATION_WEIGHTS`):

| strukturell | Gewichte / Feinabstimmung |
|---|---|
| `ADD_NODE`, `REMOVE_NODE` | `MOD_WEIGHT` (±0.1) |
| `ADD_CONN`, `REMOVE_CONN` | `MOD_WEIGHT_LARGE` (Reset auf ±2.0) |
| `ADD_LAYER` (ganze Hidden-Schicht) | `MOD_BIAS`, `MOD_ACTIVATION`, `SWAP_NODES` |

`eternity_deep.py` schaltet diese Gewichte adaptiv um: **EXPLORE** (strukturlastig, baut
tiefe Netze) und nach 6 erfolglosen Runden **EXPLOIT** (nur noch Gewichte/Bias). Nach jeder
Verbesserung geht es zurück zu EXPLORE.

### Ablauf einer Generation

1. Jedes Gehirn spielt `EVAL_GAMES` Spiele; der Score wird gemittelt (dämpft Zufallsrauschen).
2. Population nach Score sortieren.
3. Die besten 20 % (`ELITISM_RATE`) werden unverändert übernommen.
4. Der Rest entsteht aus Crossover zweier selektierter Eltern + Mutation.

**Temp-Elite:** Vielversprechende Nachkommen bekommen geschützte Plätze und müssen sich über
mehrere Generationen behaupten, bevor sie in die permanente Elite aufgenommen werden — sonst
werden strukturelle Mutationen bestraft, bevor sie sich beweisen können.

**Longevity:** Jedes Netz zählt, wie viele Generationen es ununterbrochen in der Elite war.
Der „Veteran" ist das Netz mit der höchsten je gesehenen Longevity — nicht zwingend das mit
dem höchsten Einzelscore.

---

## Trainings-Workflow

### `train.py` — mit Live-Ansicht

600 Generationen à 50 Spiele, alle Spiele parallel im Fenster sichtbar. Die Spiellogik läuft
ungedrosselt, gerendert wird mit festen 60 FPS. Am Ende: matplotlib-Fenster mit 2×2-Graphen
sowie gespeicherter Veteran, Elite-Seeds und CSV-Report.

Konfiguration im Skript-Kopf (`GENERATIONS`, `LEVEL`, `PROFILE_NAME`, `SCORING_MODE`,
`SELECTION_STRATEGY`, `ELITISM_RATE`, `SEED_PATH`).

### `eternity.py` / `eternity_deep.py` — Dauertraining

Selbstverbessernde Endlosschleife in zwei Phasen:

**Phase 1 · Bootstrap** — 300 Generationen von Null. Bestanden, wenn der beste Score in den
letzten 5 Generationen über `BOOTSTRAP_THRESHOLD` liegt; sonst von vorne.

**Phase 2 · Evolution** — 300 Generationen mit den Elite-Seeds des letzten erfolgreichen
Laufs. Bestanden, wenn der Median um mindestens `IMPROVEMENT_FACTOR` (1.05) steigt.

- Erfolg → Seed aktualisieren, weiter
- Fehlschlag → gleicher Seed nochmal, Fehlerzähler hoch
- Nach `MAX_FAILURES` Fehlschlägen in Folge → Paket speichern, zurück zu Phase 1

Nach jedem Bootstrap-Erfolg und jeder Verbesserung wird ein Checkpoint geschrieben, das
Training ist also jederzeit unterbrechbar. `eternity_deep.py` fragt beim Start, ob neu
gestartet, ein Seed geladen oder ein Checkpoint fortgesetzt werden soll — und ob der
JIT- oder der Python-Pfad verwendet wird.

### Typischer Zyklus

```
eternity_deep.py  →  runs/eternity_deep/<timestamp>_run<N>/
        ↓
plot_csv.py             Verlauf prüfen
inspect_network.py      Topologie anschauen
play.py                 Verhalten beurteilen
        ↓
Seed aus dem Paket zurück in eternity_deep.py (Startmenü, Option 2)
```

---

## Ausgabedateien

| Datei | Inhalt |
|---|---|
| `veteran_*.pkl` | `{"network": ..., "profile_name": ..., "longevity": ...}` |
| `elite_*.pkl` | Liste der Top-Netze — Warm-Start-Seed für den nächsten Lauf |
| `training_report_*.csv` | eine Zeile pro Generation, Score-Breakdown |
| `progress.csv` | wie oben, aber laufübergreifend fortgeschrieben |
| `eternity_summary.csv` | eine Zeile pro gespeichertem Paket, nach Median sortiert |

Spalten der Report-CSV:

```
generation, best_score, mean_score, median_score, worst_score,
best_turns, median_turns, worst_turns, best_food_eaten,
best_sc_survival, best_sc_towards, best_sc_against, best_sc_ate, best_sc_bomb
```

`progress.csv` enthält zusätzlich `elite_median_score`.

---

## Performance

`evolution/fast_eval.py` ist eine Numba-JIT-Variante des kompletten inneren Spielloops
(Sensoren + Forward-Pass + Spielschritt), die Netze in CSR-Arrays umwandelt. Fällt
automatisch auf den Python-Pfad zurück, wenn `numba` fehlt.

Weil `@njit`-Funktionen das GIL freigeben, nutzt `eternity_deep.py` einen `ThreadPool` statt
`multiprocessing.Pool` — echte Parallelität ohne Pickling der Gehirne und ohne IPC-Overhead.
`eternity.py` (Python-Pfad) nutzt weiterhin echte Prozesse.

Größenordnung: ~50× schneller pro Evaluation im JIT-Pfad, ~10 ms statt ~28 ms pro Generation
bei 50 Gehirnen. Details in [`docs/update.md`](docs/update.md).

Eine GPU bringt hier nichts: variable Topologien lassen sich nicht sinnvoll batchen und der
Spielloop ist inhärent sequenziell.

---

## Tests

45 Unit-Tests für Sensoren, Profile und Mutationsoperatoren:

```bash
python tests/test_sensors.py
python tests/test_mutations.py
```

---

## Technische Hinweise

**`sys.path`-Struktur.** Die Skripte legen `evolution/` per `sys.path.insert` auf den Pfad und
importieren die Module flach (`from network import ...`, nicht `from evolution.network import ...`).
Das ist historisch gewachsen — **und muss so bleiben:** die gespeicherten `.pkl`-Dateien
referenzieren die Klassen als `network.Network` bzw. `network.Node`. Würde `evolution/` zu
einem echten Package mit relativen Imports umgebaut, wären sämtliche gespeicherten Netze
nicht mehr ladbar.

**Zwei `constants.py`.** `game/constants.py` (Spielregeln) und `evolution/constants.py`
(GA-Konfiguration) sind verschiedene Module. Durch die flachen Imports meint
`from constants import config` immer das aus `evolution/`.

**Konstanten-Duplikat.** `evolution/fast_eval.py` spiegelt die Werte aus `game/constants.py`
als Modulkonstanten (`_FS`, `_FOOD_STEPS`, …), weil Numba nicht auf Python-Objekte zugreifen
kann. Ändert sich eine Spielkonstante, muss sie dort mitgezogen werden.

**`cg_base/`** ist das ursprüngliche C/OpenGL-Projekt. Es liegt lokal als Referenz, ist aber
per `.gitignore` bewusst nicht versioniert.

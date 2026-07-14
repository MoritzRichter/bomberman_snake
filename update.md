# Update-Log — Bomberman Snake ML

Alle größeren Änderungen aus dieser Konversation in chronologischer Reihenfolge.

---

## 1. Konsolenmüll behoben — `evolution/constants.py`
`self.warnings = False` gesetzt. Beendet den Spam von „no nodes to remove"-Meldungen bei jeder neuen Generation.

---

## 2. Score-Komponenten — `evolution/agent.py`
Jeder Agent trackt jetzt einzelne Score-Anteile:
- `score_survival` — kleiner Bonus pro überlebtem Schritt
- `score_towards` / `score_against` — Bewegung auf Essen zu / weg
- `score_ate` — Essen gefressen
- `score_bomb` — Strafe wenn Bombe explodiert
- `food_eaten` — Zähler

---

## 3. Neue Score-Parameter — `evolution/constants.py`
```
points_survived_tick = 0.01
points_bomb_exploded = 0
points_against_food  = -0.5   (gesenkt)
```

---

## 4. Trainingschleife entkoppelt — `train.py`
Spiellogik läuft jetzt **so schnell wie möglich**, Anzeige wird mit festem 60-FPS-Takt gerendert. Time-Budget-Loop: in jedem 16-ms-Frame werden so viele Ticks wie möglich abgearbeitet, dann ein Render-Frame.

---

## 5. Score-Breakdown in Konsole & CSV — `train.py`
Pro Generation zwei Zeilen Ausgabe:
- Zeile 1: Gesamt-Score, Median, Mean, Worst + Turns + Veteran-Longevity
- Zeile 2: `└ best [food / turns]: surv +X (Y%) →food +X ate +X ←food bomb`

CSV-Felder erweitert um:
`best_food_eaten`, `best_sc_survival`, `best_sc_towards`, `best_sc_against`, `best_sc_ate`, `best_sc_bomb`

---

## 6. Veteran-Tracking — `train.py`
- Jedes Gehirn hat ein `longevity`-Attribut (wie viele Generationen es durchgehend zur Elite gehörte).
- Elitisten: `longevity += 1` pro Generation; neue Offspring: `longevity = 0`.
- `veteran_brain` zeigt immer auf das Objekt mit der höchsten gesehenen Longevity.
- Am Ende wird das Veteran-Gehirn als `veterans/veteran_TIMESTAMP.pkl` gespeichert.

---

## 7. Seed-System — `train.py`
- Am Ende jedes Trainings werden die Top-ELITISM-Gehirne als `seeds/elite_TIMESTAMP.pkl` gespeichert.
- `SEED_PATH` in den Config-Konstanten: wird geladen und als Startpopulation verwendet (Warm-Start für den nächsten Run).
- Inkompatible Gehirne (falscher `input_size`) werden automatisch herausgefiltert.

---

## 8. End-of-Training-Graphen — `train.py`
Nach Trainingsende: `matplotlib`-Fenster mit 2×2-Subplots (Dark Background):
- Score-Übersicht (Best / Median / Mean / Worst)
- Überlebte Turns + Essen gefressen (Balkendiagramm)
- Score-Breakdown Gewinne (Survival / Towards / Ate)
- Score-Breakdown Strafen (Against / Bomb)

`KeyboardInterrupt` beim Schließen des Fensters abgefangen (Windows/Tk-Konflikt).

---

## 9. CSV-Speicherpfad — `train.py`
Training-Reports werden jetzt in `Training_Report/training_report_TIMESTAMP.csv` gespeichert (Unterordner neben `train.py`, wird automatisch angelegt).

---

## 10. Tuner aktualisiert — `tuner.py`
`TrialConfig` hat die neuen Felder `score_survived_tick` und `score_bomb_exploded`. Beide werden in `_run_trial` korrekt gesetzt/wiederhergestellt und in `as_dict()` exportiert.

---

## 11. `play.py` — neues Programm
Lädt einen gespeicherten Veteranen und lässt ihn spielen. Features:
- Auswahlbildschirm: Veteranen-Liste (neueste zuerst), Level-Buttons 1/2/3, Scrollbar
- Info-Panel: food_eaten, turns, level, profile, longevity, Dateiname
- **Gamegeschwindigkeit mit ↑/↓ anpassbar**: `[0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]` steps/s
  - ≤ 60 TPS: Timer-basiert (max. 1 Tick pro Frame)
  - \> 60 TPS: Frame-Budget (so viele Ticks wie in 14 ms passen)
- Steuerung: R = Neustart, ESC = Auswahl, Q = Beenden
- Alle Pfade absolut (kein CWD-Problem)

---

## 12. `plot_csv.py` — neues Programm
Plottet jeden Training-Report-CSV mit denselben 2×2-Graphen wie `train.py`.
- Aufruf ohne Argument: öffnet nativen Datei-Dialog **im Skript-Ordner**
- Aufruf mit Pfad-Argument: direkt
- Alte CSVs (ohne Breakdown): automatisch 2×1-Fallback
- Terminale Fallback-Auswahl sucht rekursiv im Skript-Ordner nach `training_report_*.csv`

---

## 13. `eternity.py` — neues Programm
Dauerhafter, selbstverbessernder Trainingsloop ohne Rendering.

### Phase 1 — Bootstrap
- 300 Generationen von Null starten
- Prüfung: `max(best_score in letzten 5 Gens) > 1000`
- Scheitert → von vorne

### Phase 2 — Evolution
- 300 Generationen mit Elite-Seeds des letzten erfolgreichen Runs
- Prüfung: `median(letzten 5 Gens) ≥ vorheriger_median × 1.05`
- Erfolg → Seed aktualisieren, `failure_count = 0`, weiter
- Scheitern → gleichen Seed nochmal, `failure_count += 1`
- Nach 15 aufeinanderfolgenden Fehlern → Eternity-Paket speichern, Phase 1 neu starten

### Ausgabe pro gespeichertem Paket (`Eternity-Run/TIMESTAMP_runNNN/`)
- `veteran_TIMESTAMP.pkl`
- `elite_TIMESTAMP.pkl`
- `training_report_TIMESTAMP.csv` (letzter verbesserter Run)

### `Eternity-Run/eternity_summary.csv`
Nach jedem gespeicherten Paket wird die globale Rangliste aller Runs neu geschrieben, sortiert nach `final_median` (absteigend). Spalten:
`rank, run_index, timestamp, final_median, successful_improvements, evo_rounds, boot_attempts, folder`

### Parallelisierung (Multiprocessing)
Alle 50 Agenten pro Generation werden parallel in einem `multiprocessing.Pool` ausgewertet — Speedup ≈ Anzahl CPU-Kerne. Pool wird einmal gestartet und für den gesamten Lauf wiederverwendet.
- `N_WORKERS = None` → alle logischen Kerne
- `N_WORKERS = 1` → sequenziell (kein Pool, für Debugging)

---

## 14. Stabile Perm-Elite + TempElite-System — `eternity.py`

### Stabiler Elite-Pool
- Gen 0: Top-ELITISM-Gehirne werden einmalig per Score-Ranking gesetzt (`perm_elite_set`)
- Gen 1+: `perm_elite_set` ist eingefroren — kein Re-Ranking, kein automatischer Austausch
- Änderung nur über TempElite-Promotion möglich

### TempElite-Mechanismus
- Nicht-Elite-Gehirne die `prev_worst_elite_score` (letzter Gen schlechtester Perm-Elite) schlagen → Temp-Pool
- Müssen **jede Runde** den Threshold übertreffen (sonst eliminiert)
- Nach `TEMP_PROMOTE_AFTER = 3` erfolgreichen Runden → ersetzt schlechtesten Perm-Elite
- Max. `MAX_TEMP_ELITE = 5` geschützte Slots gleichzeitig

### VERBOSE-Flag
`VERBOSE = False` — TempElite-Promotionsmeldungen standardmäßig stumm; auf `True` für Debug-Output.

---

## 15. Bomben-Malus aktiviert — `evolution/constants.py`
`points_bomb_exploded = -5` (war `0`). Schlange lernt jetzt aktiv, Bomben zu verhindern.

---

## 16. Kontinuierliche Richtungs-Sensoren — `evolution/sensors.py`
`is_food_forward/left/right` und `is_bomb_forward/left/right` geben jetzt Distanz-basierte Float-Werte zurück statt binär 0/1:
- Formel: `1.0 - dist / FIELDSIZE` (Abstand 1 → 0.9, Abstand 9 → 0.1, falsche Richtung → 0.0)
- `can_move_*` bleibt binär (sicher/nicht sicher)
- `SENSOR_FUNCS`-Wrapper `1 if ... else 0` entfernt

---

## 17. Body-Proximity-Sensoren — `evolution/sensors.py` + `profiles.py`
Drei neue Sensoren: `body_forward`, `body_left`, `body_right`
- Wirft einen Strahl entlang der jeweiligen Achse und findet das nächste eigene Körpersegment
- Gleiche Distanz-Formel wie Richtungs-Sensoren; 0.0 = kein Körper auf der Achse
- Gewicht 1.5 in `bomb_aware`, `timer`, `full` (höher gewichtet als Standard, da Körperkollision häufigste Todesursache)
- Profile-Inputzahlen: `bomb_aware` 9→12, `timer` 12→15, `full` 14→17

---

## 18. Mutations-Gewichte — `evolution/evolution.py`

### Gewichtete Mutations-Auswahl
`random.choice` → `random.choices` mit `MUTATION_WEIGHTS`. Parametrische Mutationen 3× häufiger als strukturelle:
- Strukturell (ADD/REMOVE Node/Conn): Gewicht 1 je, außer ADD_CONN: 2
- Parametrisch (MOD_WEIGHT, MOD_BIAS, SWAP_NODES): Gewicht 3–4
- MOD_ACTIVATION: Gewicht 2 (disruptiv wie strukturell)

### Neuer Mutations-Typ: MOD_WEIGHT_LARGE
- `MOD_WEIGHT` (Gewicht 4): addiert `±0.1` auf ein Gewicht (Feintuning, war ±0.5)
- `MOD_WEIGHT_LARGE` (Gewicht 1): **setzt** Gewicht auf `uniform(−2.0, +2.0)` (Reset, hilft aus lokalen Optima)
- `constants.py`: `connectionWeight` ±0.1, neues `connectionWeightLarge` ±2.0

---

## 19. Multi-Game-Evaluation + Level-Rotation — `eternity.py`

### Multi-Game (EVAL_GAMES = 3)
Jedes Gehirn spielt 3 Spiele pro Generation; Score und alle Stats werden gemittelt. Reduziert Zufalls-Rauschen durch zufälliges Food-Placement und Bombe.

### Level-Rotation (LEVEL_ROTATION = False)
Opt-in: wenn `True`, spielt jedes Gehirn ein Spiel auf Level 1, 2 und 3 statt dreimal dasselbe Level. Produziert robustere Agenten die nicht nur ein Layout auswendig lernen.

---

## 20. Tournament-Selection — `eternity.py`
Konfigurierbare Eltern-Selektion neben dem bisherigen Power-Modus:
- `SELECTION_STRATEGY = "power"` (Standard) — exponentieller Bias zu Top-Rängen
- `SELECTION_STRATEGY = "tournament"` — K zufällige Kandidaten, besten nehmen
- `SELECTION_POWER = 4` — Stärke des Power-Bias
- `TOURNAMENT_K = 5` — Kandidatenanzahl pro Tournament

Tournament-Selection ist bereits in `selection.py` implementiert; `eternity.py` übergibt nun `power=` und `k=` an alle `getOffspring`-Aufrufe.

---

## 21. Kumulativer Fortschritts-CSV — `eternity.py`
`_append_progress(results, gen_offset)` schreibt nach jedem erfolgreichen Bootstrap und jeder Evo-Verbesserung die Ergebnisse an `Eternity-Run/checkpoint/progress.csv` an. Generationsnummern sind kumulativ verschoben (`gen_offset`), sodass der Gesamtfortschritt über alle Phasen hinweg in einer einzigen CSV sichtbar ist (Bootstrap = Gen 1–300, erste Verbesserung = 301–600 usw.).

---

## 22. Visuelle Best-Brain-Demo — `eternity.py`
`VISUAL = False` Config-Flag. Wenn aktiviert: zwischen den Generationen spielt der beste Agent eine Demo-Runde im Pygame-Fenster.

- `_build_obs(game)` — baut ein 2D-Grid ohne numpy für das Rendering
- `_run_visual_demo(brain, visual_ctx, header)` — rendert ein komplettes Spiel mit Tick-Budget pro Frame
- `run_training()` bekommt `visual_ctx`-Parameter und ruft `_run_visual_demo(sorted_brains[0], ...)` nach jeder Generation auf
- `main()` erstellt das Pygame-Fenster einmalig und gibt es als `visual_ctx`-Dict weiter; `pygame.quit()` im `finally`-Block

Multiprocessing-Evaluation läuft headless und unberührt; das Fenster ist nur zwischen Generationen aktiv.

---

## 23. Eternity-Graphen — `eternity.py`
`show_eternity_graphs()` liest den kumulativen `progress.csv` und zeigt ein 4-Panel-Matplotlib-Diagramm (Dark Background):
- Score-Übersicht (Best / Median / Elite-Median / Mean / Worst)
- Turns + Food-Balkendiagramm
- Score-Gewinne (Survival / Towards / Ate)
- Score-Strafen (Against / Bomb)

Gestrichelte vertikale Linien markieren jede erfolgreiche Phasengrenze. Aufruf automatisch nach Package-Speicherung und bei `KeyboardInterrupt`.

---

## 24. `inspect_network.py` — neues Programm
Interaktiver Pygame-Visualisierer für gespeicherte Veteranen-Netzwerke.

### Layout
- Fenster 1400×840, Canvas links (1100px), Sidebar rechts (300px)
- **BFS Longest-Path**: Tiefe jedes Nodes wird aus der Graphstruktur berechnet → Spalten
- Output-Nodes immer in der letzten Spalte; nicht erreichbare Nodes ebenfalls dort

### Interaktion
- **Pan**: Linke Maustaste ziehen
- **Zoom**: Scrollrad (zentriert auf Cursor)
- **Klick**: Trifft Node oder Verbindung → Selektion + Sidebar-Details
- **T / Shift+T**: Gewichtsschwelle heben/senken (schwache Verbindungen ausblenden)
- **L**: Gewichts-Labels an/aus
- **R**: Kamera zurücksetzen
- **ESC / Q**: Zurück zur Dateiauswahl

### Darstellung
- Input-Nodes blau, Hidden orange, Output grün, Selektion gelb
- Verbindungsfarbe: grün (positiv) / rot (negativ); Interpolation von `C_WEAK=(110,110,135)` zu Zielfarbe (Minimum-Alpha 80 für Sichtbarkeit)
- Node-Index als kleine Zahl unterhalb jedes Nodes
- Aktivierungsfunktionsname innerhalb der Node (ab Zoom > 0.5)

### Dateiauswahl
`find_veterans()` durchsucht per `os.walk` den gesamten Projektordner nach `veteran*.pkl` (neueste zuerst). Direktpfad als CLI-Argument möglich.

### Bugfixes (nach erstem Test)
- **KeyError beim Node-Klick**: Verbindungen referenzierten Nodes außerhalb von `pos` (orphaned connections nach Mutationen). Fix: `if id(c.from_node) not in pos: continue` in `draw_network` und `hit_conn`
- **Schlechter Kontrast**: Farb-Interpolation startete bei `BG=(18,18,26)` → schwache Verbindungen unsichtbar. Fix: Interpolation von `C_WEAK=(110,110,135)`, Minimum-Alpha = 80

---

## 25. `eternity_deep.py` — neues Programm
Kopie von `eternity.py` mit adaptiver Mutations-Strategie für tiefere Netzwerke. Ausgabe nach `Eternity-Deep/`.

### Unterschiede zu `eternity.py`

| Parameter | eternity.py | eternity_deep.py | Grund |
|---|---|---|---|
| `SELECTION_POWER` | 4 | 2 | Strukturelle Mutanten überleben länger |
| `PROFILE_NAME` | `"raw"` | `"full"` | 19 Sensoren statt 108 Rohdaten |
| `BOOTSTRAP_THRESHOLD` | 1000 | 500 | Angepasst an full-Profil |
| `MAX_FAILURES` | 5 | 10 | Mehr Zeit für tiefe Nets |

### Adaptive Mutations-Gewichte
Zwei Presets, die zur Laufzeit in `evolution.MUTATION_WEIGHTS` geschrieben werden:

**EXPLORE** (Standard — strukturelles Wachstum):
- `ADD_NODE: 4`, `ADD_LAYER: 3`, `ADD_CONN: 3` — deutlich häufiger als bisher
- `MOD_WEIGHT: 2`, `MOD_BIAS: 2` — reduziert während Wachstum

**EXPLOIT** (wenn festgesteckt — Gewichtstuning):
- `MOD_WEIGHT: 6`, `MOD_BIAS: 5` — dominant
- `ADD_LAYER: 0`, `REMOVE_NODE: 0`, `REMOVE_CONN: 0` — keine strukturellen Änderungen

### Wechsel-Logik
- Standard: EXPLORE
- Nach `SWITCH_TO_EXPLOIT_AFTER = 3` aufeinanderfolgenden Evo-Fehlern → EXPLOIT
- Nach jeder Verbesserung → zurück zu EXPLORE (oszilliert)

`_apply_mutation_mode(mode)` schreibt das gewählte Preset per `_evo.MUTATION_WEIGHTS.update(weights)` direkt in das Evolution-Modul. Sicher, da `mutatePopulation` ausschließlich im Hauptprozess läuft (Worker evaluieren nur).

---

## 26. `play.py` — Veterans aus Eternity-Ordnern laden
`list_veterans()` durchsucht jetzt drei Quellen statt nur `veterans/`:
- `veterans/*.pkl`
- `Eternity-Run/**/veteran*.pkl` (rekursiv)
- `Eternity-Deep/**/veteran*.pkl` (rekursiv)

Neue Hilfsfunktion `veteran_label(path)` erzeugt kurze Labels für die Dateiliste:
- `[Vet]  datei.pkl`
- `[Run/run001]  veteran_xxx.pkl`
- `[Deep/ckpt]  veteran.pkl`

Label wird auch im Info-Panel während des Spiels angezeigt.

---

## 27. `plot_csv.py` — Simplified View
Neue Ansicht: nur die wichtigsten Metriken als geglättete Linien, keine Rohdaten sichtbar.

**Aktivierung:**
- Konstante oben: `SIMPLE_VIEW = True`
- CLI-Flag: `python plot_csv.py -s` oder `python plot_csv.py bericht.csv -s`

**Darstellung:**
- 2 Panels nebeneinander (Score links, Turns rechts)
- Best score, Median score, Elite-Median (falls vorhanden), Best turns, Median turns
- Food eaten als gestrichelte Linie auf Twin-Achse

**Glättung** (`_smooth(values, window)`):
- Rolling Average via `numpy.convolve` mit Edge-Padding
- `SMOOTH_WINDOW = 0` → automatische Fenstergröße: `max(5, len(daten) / 20)`
- Manuell auf feste Zahl setzbar (z.B. `SMOOTH_WINDOW = 20`)

---

## 28. `inspect_network.py` — Checkpoint-Veterans erkannt
`find_veterans()` filterte bisher nur `veteran_*.pkl` (mit Underscore). Checkpoint-Dateien heißen `veteran.pkl` (ohne Suffix) und wurden übersehen.

Fix: `f.startswith("veteran_")` → `f.startswith("veteran")` — erkennt jetzt beide Formate.

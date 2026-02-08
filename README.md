# Iterative Prompt Optimization for Hate Speech Detection

Begleitcode zur Bachelorarbeit **"Iterative Promptoptimierung zur Klassifikation von Hassrede"** (2025).

---

## 📖 Überblick

Dieses Framework implementiert einen **automatisierten Feedback-Zyklus** zur Optimierung von System-Prompts für Large Language Models (LLMs) im Kontext der deutschsprachigen Hate Speech Detection.

### Methodischer Ansatz

Das Framework arbeitet in **3 Phasen**:

1. **Phase 1: Initialisierung**
   - Baseline-Prompt (Zero-Shot oder manuell)
   - Stratifizierte Datenaufteilung (Train/Val/Test)
   - Baseline-Evaluation auf Validierungsdaten (Iteration 0)

2. **Phase 2: Iterative Optimierung** (Hauptschleife)
   - Inferenz auf Validierungsdaten (Google Gemma 3 27B)
   - Fehleranalyse (False Positives & False Negatives)
   - Prompt-Verbesserung durch Meta-LLM
   - Tracking des besten Prompts (S-Score als Zielfunktion)
   - **Testdaten werden NICHT berührt** (Held-Out-Set)

3. **Phase 3: Finale Evaluation**
   - Einmalige Evaluation auf unabhängigem Testdatensatz
   - Vergleich: Baseline vs. optimierter Prompt
   - Statistische Signifikanzprüfung (McNemar-Test)

---

## 🚀 Installation

### Voraussetzungen

- **Python 3.12+**
- **Poetry** (Dependency Management)
- **Ollama** (für lokale LLM-Inferenz): https://ollama.ai/

### Schritt 1: Repository klonen

```bash
git clone https://github.com/mlauberg/ba-hatespeech-framework-final.git
cd ba-hatespeech-framework-final
```

### Schritt 2: Dependencies installieren

```bash
# Poetry installieren (falls nicht vorhanden)
curl -sSL https://install.python-poetry.org | python3 -

# Abhängigkeiten installieren
poetry install

# Virtual Environment aktivieren
poetry shell
```

### Schritt 3: LLM-Modell vorbereiten

```bash
# Ollama muss laufen (separates Terminal)
ollama serve

# Modell herunterladen (einmalig, ~15 GB)
ollama pull gemma2:27b
```

---

## 📊 Verwendung

### 1. Datensatz vorbereiten

Legen Sie Ihren Datensatz im Ordner `data/` ab:

```
data/
└── raw/
    └── gutefrage.csv    # Oder HOCON34k
```

**Erforderliche Spalten:**
- `text`: Textinhalt
- `label`: Binäres Label (0 = NOT-HS, 1 = HS)

### 2. Konfiguration anpassen

In `main.py` (Zeilen 74-86):

```python
DATA_FILE = "data/raw/gutefrage.csv"    # Pfad zum Datensatz
DATASET_TYPE = "gutefrage"              # Oder "hocon"
SAMPLE_SIZE = 100                       # Stichprobengröße
MAX_ITERATIONS = 100                    # Optimierungszyklen
MAX_ERRORS_TO_ANALYZE = 10              # Fehler pro Iteration

INITIAL_PROMPT = """..."""              # Baseline-Prompt
```

### 3. Experiment starten

```bash
poetry run python main.py
```

**Ausgabe während der Ausführung:**
- Progress Bar für jede Iteration
- Metriken nach jedem Zyklus (F1, F2, MCC, S-Score)
- ETA (geschätzte Restlaufzeit)
- Automatische Speicherung der Ergebnisse

### 4. Ergebnisse auswerten

Nach Abschluss finden Sie im Ordner `results/`:

- `experiment_YYYYMMDD_HHMMSS.csv`: Metriken pro Iteration
- `experiment_YYYYMMDD_HHMMSS.json`: Vollständiger Prompt-Verlauf
- `best_prompt.txt`: Bester gefundener Prompt

---

## 📈 Metriken

Das Framework verwendet eine **kombinierte Zielfunktion** (S-Score):

### S-Score = F2-Score + MCC

- **F2-Score**: Priorisiert Recall (β=2)
  → Vermeidung von False Negatives (übersehene Hassrede)

- **MCC**: Matthews Correlation Coefficient
  → Robustheit bei Klassenungleichgewicht

**Zusätzlich berechnet:**
- F1-Score
- Accuracy
- Precision
- Recall

---

## 🏗️ Projektstruktur

```
ba-hatespeech-framework-final/
├── README.md                   # Diese Datei
├── LICENSE                     # MIT License
├── pyproject.toml              # Dependencies (Poetry)
├── .gitignore
│
├── main.py                     # Hauptschleife (3 Phasen)
│
├── src/
│   ├── __init__.py
│   ├── metrics.py              # Metrik-Berechnung (F2, MCC, S-Score)
│   ├── optimizer.py            # Error-driven Meta-Prompting
│   ├── inference.py            # LLM-Inferenz (Ollama-Integration)
│   └── data_loader.py          # Datenverarbeitung & Stratified Split
│
└── data/
    ├── .gitkeep
    └── raw/                    # Datensätze hier ablegen
```

---

## 🔬 Wissenschaftlicher Hintergrund

### Kernidee: Error-Driven Meta-Prompting

Anstatt Prompts manuell zu optimieren, analysiert ein **Meta-LLM** (derselbe Gemma 3 27B) die Fehler einer Iteration und generiert **synthetische Few-Shot-Beispiele**, um die Entscheidungsgrenze zu schärfen.

**Stateless-Design:**
- Der Optimizer hat **keinen Zugriff** auf die Historie
- Jede Iteration basiert nur auf: `aktueller Prompt + aktuelle Fehler`
- Verhindert "Prompt Bloat" durch unkontrolliertes Anhängen

### Datensätze

- **gutefrage.net**: Q&A-Plattform (informelle Sprache, Slang)
- **HOCON34k**: Zeitungskommentare (formeller Kontext)

→ Cross-Domain-Validierung zur Prüfung der Generalisierungsfähigkeit

---

## 📝 Zitation

Falls Sie diese Arbeit in Ihrer Forschung verwenden:

```bibtex
@thesis{lauberger2025,
  author = {Maximilian Lauberger},
  title = {Iterative Promptoptimierung zur Klassifikation von Hassrede},
  school = {Hochschule München, Fakultät für Informatik und Mathematik},
  year = {2025},
  type = {Bachelor's Thesis},
  url = {https://github.com/mlauberg/ba-hatespeech-framework-final}
}
```

---

## 🛠️ Technische Details

### Hardware-Anforderungen

- **GPU**: Mind. 20 GB VRAM (für Gemma 3 27B in Q4-Quantisierung)
- **RAM**: 32 GB empfohlen
- **Speicher**: ~50 GB für Modell + Datensätze

### Software-Stack

- **Python 3.12**
- **Ollama** (LLM-Inferenz-Server)
- **scikit-learn** (Metriken)
- **pandas** (Datenverarbeitung)
- **tqdm** (Progress Bar)

---

## 🤝 Beitragen

Dieses Repository dient primär der **Reproduzierbarkeit** der Bachelorarbeit.
Für Fragen oder Anregungen:

- **Autor**: Maximilian Lauberger
- **E-Mail**: lauberge@hm.edu
- **Betreuer**: Prof. Dr. Peter Mandl (Hochschule München)

---

## 📄 Lizenz

MIT License - Siehe [LICENSE](LICENSE) für Details.

---

## 🙏 Danksagung

Dank an:
- Prof. Dr. Peter Mandl für die wissenschaftliche Betreuung
- Die Entwickler von Ollama und Google Gemma für die Open-Source-Modelle
- Die Ersteller der Datensätze gutefrage.net und HOCON34k

---

## 🔗 Weiterführende Links

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Google Gemma Models](https://ai.google.dev/gemma)
- [Poetry Documentation](https://python-poetry.org/)

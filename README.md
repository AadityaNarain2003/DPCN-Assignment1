# Clean Questionnaire Network Analysis

The original six scripts contain two copies of nearly the same pipeline:
people and questions. This version separates **configuration** from
**implementation**.

## Structure

```text
network_analysis_clean/
├── config/
│   ├── people.yaml
│   └── questions.yaml
├── src/
│   ├── config.py
│   ├── data.py
│   ├── similarity.py
│   ├── network.py
│   └── plotting.py
├── scripts/
│   ├── build_people_matrix.py
│   ├── build_question_matrix.py
│   └── analyze_network.py
├── requirements.txt
└── README.md
```

## Commands

Run these from the project root:

```bash
pip install -r requirements.txt

python scripts/build_people_matrix.py
python scripts/build_question_matrix.py

python scripts/analyze_network.py people
python scripts/analyze_network.py questions
```

## Where to change things

Use `config/people.yaml` for participant-specific settings and
`config/questions.yaml` for question-specific settings.

That includes paths, overlap thresholds, network thresholds, ER simulation
count, random seed, display settings, question domains, and domain styles.

The shared Python code should generally not need to change when you change
an analysis parameter.

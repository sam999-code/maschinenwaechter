# maschinenwaechter — نظام الصيانة التنبؤية (مشروع تعلّم طويل الأمد)

> **Deutsch:** *Maschinenwächter* = „Maschinen-Wächter" — ein Lernprojekt für Predictive-Maintenance-SaaS für den deutschen Mittelstand.
> **English:** *Machine Guard* — a long-term learning project: a predictive-maintenance SaaS for the German SME (Mittelstand) market.

---

## ما هي الصيانة التنبؤية؟ / What is predictive maintenance?

**بالعربية:** الصيانة التنبؤية تعني مراقبة آلات المصانع بأجهزة استشعار (الاهتزاز، الحرارة، الصوت، سرعة الدوران) والتنبّؤ بالأعطال **قبل** حدوثها، بدلاً من الصيانة الدورية (تغيير القطع بلا حاجة) أو الصيانة عند الكسر (توقّف خط الإنتاج). نموذج يتعلّم الإشارات الطبيعية للآلة، ثم يُنذر عندما تنحرف هذه الإشارات.

**Why Germany's Mittelstand?** Germany's ~3.5 million SMEs (der Mittelstand) run the factories that power Europe's industry. An unplanned bearing failure on a CNC line can idle a whole plant; most Mittelstand firms have no data-science team to build monitoring in-house. A **simple, affordable, privacy-first (DSGVO)** monitoring SaaS is exactly the product this market needs — and exactly the product this project learns to build.

**Deutsch:** Predictive Maintenance = Maschinendaten überwachen und Ausfälle **vorher** erkennen — weder planlose Zeitintervalle noch Reaktion nach dem Schaden. Für den Mittelstand: einfach, bezahlbar, DSGVO-konform.

---

## خارطة الطريق: 8 مراحل / Roadmap: 8 Phases / Roadmap: 8 Phasen

| Phase | المحتوى / Content | الحالة / Status |
|---|---|---|
| 1 | محاكي بيانات أجهزة الاستشعار (اهتزاز، حرارة، صوت، RPM + حقن عطل) — Sensor simulator | ✅ مكتمل / done |
| 2 | هيكل المشروع الاحترافي + الاختبارات — Structure, packaging, pytest | ✅ مكتمل / done |
| 3 | خط الأساس الكلاسيكي: Rolling Z-Score (numpy/pandas فقط) — Classical baseline | ✅ مكتمل / done |
| 4 | نماذج عميقة بـ PyTorch: Autoencoder / LSTM لكشف الشذوذ — Deep anomaly detection | 🔜 قادم |
| 5 | التنبؤ بالعمر المتبقي للآلة (RUL) — Remaining Useful Life | 🔜 قادم |
| 6 | خدمة FastAPI + MLOps (تتبّع النماذج، إعادة التدريب) — Serving & MLOps | 🔜 قادم |
| 7 | SaaS متعدد المستأجرين + الامتثال لـ DSGVO — Multi-tenant SaaS & GDPR | 🔜 قادم |
| 8 | الإطلاق وملف الأعمال (Portfolio) — Launch & portfolio | 🔜 قادم |

## بنية المشروع / Architecture / Architektur

```
┌──────────────────────────────────────────────────────────────┐
│                        maschinenwaechter                      │
│                                                               │
│  ┌──────────────────┐        ┌───────────────────────────┐    │
│  │ simulation/      │        │ detection/                │    │
│  │  machine.py      │───────▶│  baseline.py (Phase 3)    │    │
│  │  physics-based   │  CSV   │  RollingZScoreDetector    │    │
│  │  sensor stream   │        │  (numpy + pandas only)    │    │
│  └──────────────────┘        └───────────────────────────┘    │
│           ▲                              │                    │
│  config/settings.yaml                    ▼                    │
│  sampling, windows, thresholds    reports/figures/*.png       │
│                                    data/sample/*.csv          │
│                                                               │
│  [Phase 4+]  torch Autoencoder/LSTM   [Phase 6+]  FastAPI     │
│  [Phase 5]   RUL regression           [Phase 7+]  multi-tenant│
└──────────────────────────────────────────────────────────────┘
```

## التشغيل / How to run / Ausführen

From the project root (لا حاجة لتثبيت الحزمة — no `pip install` needed; the tests and scripts bootstrap `src/` onto `sys.path`):

```bash
# 1) Generate 3 sample machines (72 h @ 1 Hz) + figures
python scripts/01_generate_data.py

# 2) Or generate one machine via the CLI
python -m maschinenwaechter.cli simulate \
    --hours 72 --hz 1 --fault-at 40 --seed 42 \
    --out data/sample/machine_01.csv \
    --plot reports/figures/machine_01.png

# 3) Run the test suite
python -m pytest tests/ -v

# 4) Explore the notebook
jupyter notebook notebooks/01_signal_exploration.ipynb
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `pyyaml`, `pytest` (all preinstalled or trivially installable). `torch`, `scikit-learn`, `fastapi`, `uvicorn` are **deliberately excluded** until Phases 4+ (see `pyproject.toml`).

## بنية الملفات / Project layout

```
maschinenwaechter/
├── config/settings.yaml          # sampling rates, rolling window, z threshold, fault config
├── src/maschinenwaechter/
│   ├── config.py                 # YAML -> dataclass settings loader
│   ├── simulation/machine.py     # THE CORE: physics-inspired simulator
│   ├── detection/baseline.py     # RollingZScoreDetector (pure numpy/pandas)
│   └── cli.py                    # `python -m maschinenwaechter.cli simulate ...`
├── scripts/01_generate_data.py   # batch generation for the sample fleet
├── tests/                        # behavioural contract tests (pytest)
├── notebooks/01_signal_exploration.ipynb
├── data/sample/                  # generated CSVs (.gitkeep'd)
└── reports/figures/              # generated PNGs (.gitkeep'd)
```

## ملاحظات تعليمية / Teaching notes

- **Phase 3 is pure numpy/pandas on purpose.** The rolling z-score is simple enough to read in one sitting, which makes it the perfect yardstick: any PyTorch autoencoder in Phase 4 must *beat this baseline* to earn its place.
- **Why `max` across sensors?** Each sensor's z-score is unit-free, so the maximum is a valid "any sensor screams" fusion rule — see the long docstring in `detection/baseline.py`.
- **Why impulsive shocks in the simulator?** Real bearing defects produce impact bursts. A rolling statistic absorbs slow drifts; it *cannot* hide sudden impulses. The simulator is built so the baseline has something legitimate to detect.
- The detector intentionally ignores `rpm`: load changes swing rpm legitimately, and an alarm keyed to production scheduling is useless.

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
| 4 | نماذج عميقة بـ PyTorch: Autoencoder / LSTM لكشف الشذوذ — Deep anomaly detection | ✅ مكتمل / done |
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
│  │  machine.py      │───────▶│  baseline.py  (Phase 3)   │    │
│  │  physics-based   │  CSV   │  autoencoder.py (Phase 4) │    │
│  │  sensor stream   │        │  + dataset.py / train.py  │    │
│  └──────────────────┘        └───────────────────────────┘    │
│           ▲                              │                    │
│  config/settings.yaml                    ▼                    │
│  sampling, windows, thresholds    evaluation/metrics.py       │
│                                   reports/figures/*.png       │
│                                   data/sample/*.csv           │
│                                                               │
│  [Phase 5]   RUL regression           [Phase 6+]  FastAPI     │
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

Dependencies: `numpy`, `pandas`, `matplotlib`, `pyyaml`, `pytest` (all preinstalled or trivially installable). `torch`, `scikit-learn`, `fastapi`, `uvicorn` are **deliberately excluded** from the base environment until Phases 4+ (see `pyproject.toml`).

**Phase 4 environment (torch):** create the project venv once, then use it for the deep-learning code and its tests (`.venv/` is gitignored):

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install scikit-learn pytest pyyaml numpy pandas matplotlib

# Run the Phase 4 experiment + the full test suite (incl. torch tests)
.venv/Scripts/python.exe scripts/04_compare_models.py
.venv/Scripts/python.exe -m pytest tests/ -v
```

The torch tests in `tests/test_autoencoder.py` are guarded with `pytest.importorskip("torch")`, so the suite also passes without the venv.

## بنية الملفات / Project layout

```
maschinenwaechter/
├── config/settings.yaml          # sampling rates, rolling window, z threshold, fault config
├── src/maschinenwaechter/
│   ├── config.py                 # YAML -> dataclass settings loader
│   ├── simulation/machine.py     # THE CORE: physics-inspired simulator
│   ├── detection/baseline.py     # RollingZScoreDetector (pure numpy/pandas)
│   ├── detection/autoencoder.py  # LSTMAutoencoder (Phase 4, torch)
│   ├── detection/dataset.py      # WindowedDataset: sliding windows (Phase 4)
│   ├── detection/train.py        # train/score/save-load the autoencoder (Phase 4)
│   ├── evaluation/metrics.py     # point-adjusted P/R/F1 + oracle threshold
│   └── cli.py                    # `python -m maschinenwaechter.cli simulate ...`
├── scripts/01_generate_data.py   # batch generation for the sample fleet
├── scripts/04_compare_models.py  # Phase 4 experiment: baseline vs. LSTM-AE
├── tests/                        # behavioural contract tests (pytest)
├── notebooks/01_signal_exploration.ipynb
├── data/sample/                  # generated CSVs (.gitkeep'd)
└── reports/figures/              # generated PNGs (.gitkeep'd)
```

## المرحلة 4: كاشف الشذوذ العميق — LSTM-Autoencoder / Phase 4: Deep anomaly detection

**الفكرة بالعربية:** الـ Autoencoder شبكة تتعلّم إعادة بناء إشارتها الخاصة. ندرّبها **فقط** على نوافذ زمنية (T=30 ثانية) من بيانات الآلة السليمة (`fault == 0`)، فيتعلّم الشكل الطبيعي للإشارة. عندما يبدأ عطل تآكل المحور (اهتزاز متنامٍ + صدمات نبضية + ارتفاع حرارة)، لا يستطيع النموذج إعادة بناء هذا النمط الغريب، فيقفز **خطأ إعادة البناء** — وهذا هو درجة الشذوذ. الفرق الجوهري عن خط الأساس في المرحلة 3: الـ z-score ينظر لكل نقطة وحيدة، أما الـ LSTM فيقرأ النافذة خطوة بخطوة ويضغط تطوّر الإشارة الزمني في متجه كامن (hidden 32)، فيميّز بين النبضة الصحية (تغيّر الحمل) والنبضة المريضة (تآكل المحامل) لأن **شكلهما الزمني مختلف** حتى لو تشابهتا في حجمها.

**Architecture:** encoder LSTM (3→32) compresses each 30×3 window to a latent vector; a decoder LSTM replays the window from that vector; a linear head projects back to 3 sensor channels. Reconstruction MSE per window = anomaly score, aligned to the window centre per timestamp (warmup edges filled ffill/bfill).

**مبدأ التدريب على السليم فقط / Train-on-healthy:** labelled faults are rare and expensive; healthy data is abundant. The model only memorises "normal", so anything abnormal reconstructs poorly — the classic semi-supervised setup for industrial anomaly detection.

### نتائج المقارنة الصادقة / Honest comparison (scripts/04_compare_models.py)

Experimental design: train machine (seed 7) ≠ test machine (seed 11) — real generalization; both detectors' thresholds calibrated on the **test machine's healthy first 35 h** only (99.5th percentile, no labels); point-adjusted P/R/F1 (segment recall, point-wise precision). Runtime ≈ 2 min on CPU.

| method | precision | recall | F1 | false alarms / healthy hour |
|---|---|---|---|---|
| Rolling z-score (Phase 3) | 0.699 | 1.000 | 0.823 | 17.7 |
| LSTM-Autoencoder (Phase 4) | **0.994** | 1.000 | **0.997** | 18.2 |

**Reading the table honestly:** both detectors find the fault segment (recall 1.0 — the z-score keys on the impulsive shocks). The difference is *precision*: ~30 % of the baseline's alarms sit in healthy regions (healthy vibration spikes look locally like shocks to a point-wise statistic), while the autoencoder's temporal context suppresses almost all of them. The raw false-alarm *rates* look high for both because a sample-wise 99.5th-percentile threshold at 1 Hz statistically must flag ~0.5 % of healthy samples — in production you would alarm on *runs* of flagged samples, not single ones. Figures: `reports/figures/phase4_comparison.png`, `reports/figures/phase4_training_loss.png`; metrics: `reports/phase4_metrics.csv`.

## ملاحظات تعليمية / Teaching notes


- **Phase 3 is pure numpy/pandas on purpose.** The rolling z-score is simple enough to read in one sitting, which makes it the perfect yardstick: any PyTorch autoencoder in Phase 4 must *beat this baseline* to earn its place.
- **Why `max` across sensors?** Each sensor's z-score is unit-free, so the maximum is a valid "any sensor screams" fusion rule — see the long docstring in `detection/baseline.py`.
- **Why impulsive shocks in the simulator?** Real bearing defects produce impact bursts. A rolling statistic absorbs slow drifts; it *cannot* hide sudden impulses. The simulator is built so the baseline has something legitimate to detect.
- The detector intentionally ignores `rpm`: load changes swing rpm legitimately, and an alarm keyed to production scheduling is useless.
- **Phase 4 must beat the Phase 3 baseline to earn its place — and on this simulator it does** (F1 0.997 vs 0.823), mainly on precision: temporal context tells a healthy load spike from a bearing shock. The honest caveats live in the Phase 4 section above.
- **Standardisation is not optional** for the autoencoder: vibration (~1 mm/s), temperature (~60 °C) and acoustic (~70 dB) differ by two orders of magnitude, so raw MSE would only ever "see" the acoustic channel. `detection/train.py` fits the scaler on healthy data and persists it next to the weights.
- **The seeds are the test suite.** Determinism (`torch.manual_seed` + `numpy` seed, seeded DataLoader generator) is what makes tests like "same seed → same first-epoch loss" possible. Never train without pinning them.

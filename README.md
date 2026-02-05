# FlightMind

An aviation-native language model built from scratch — every step documented.

Inspired by Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat), FlightMind is a transformer language model that understands aviation: regulations, weather, aircraft performance, ATC communications, and flight operations. The model also retains general world knowledge through mixed pretraining with FineWeb-EDU.

## Project Status

**Phase: Data Collection** (Local — Dell 7920, Dual Xeon Gold 5118, 64GB RAM)

## Architecture

```
Local Machine (Dell 7920)          Cloud GPU Node (8x H100)
┌──────────────────────────┐       ┌────────────────────────┐
│ Phase 1: Data Collection │       │ Phase 5: Pretraining   │
│ Phase 2: Data Cleaning   │──────>│ Phase 6: Midtraining   │
│ Phase 3: Tokenizer       │ upload│ Phase 7: SFT / DPO     │
│ Phase 4: Tokenization    │       └──────────┬─────────────┘
│ Phase 8: Evaluation      │<─────────────────┘ download
│ Phase 9: Inference/Demo  │       checkpoints
└──────────────────────────┘
```

## Data Sources

| Source | Type | Est. Tokens | Status |
|--------|------|-------------|--------|
| FAA Handbooks (PHAK, AFH, IFH...) | Technical manuals | ~5M | Scripted |
| 14 CFR (Federal Aviation Regulations) | Regulations | ~10M | Scripted |
| NTSB Accident Reports | Narratives + data | ~50M | Scripted |
| NASA ASRS Safety Reports | Pilot narratives | ~200M | Scripted |
| Historical METAR/TAF | Weather obs | ~50M | Scripted |
| ATC Transcripts (ATCO2, UWB-ATCC) | Communications | ~5M | Scripted |
| Wikipedia Aviation Articles | Encyclopedia | ~20M | Scripted |
| FAA Aircraft Registry | Structured data | ~5M | Scripted |
| OpenAP Performance Models | Aircraft data | ~1M | Scripted |
| NASA/NACA Technical Reports | Research papers | ~500M | Planned |
| SKYbrary Safety Articles | Knowledge base | ~10M | Planned |
| Military Flight Manuals | Technical manuals | ~100M | Manual DL |
| FineWeb-EDU (general knowledge) | Web text | 1.3T available | Cloud |

## Model

- **Architecture**: Transformer (nanochat-style, depth-parameterized)
- **Target size**: d20 (~561M params) or d32 (~1.75B params)
- **Tokenizer**: Custom BPE (32K vocab) trained on aviation + general text
- **Training data**: 70% general (FineWeb-EDU) + 30% aviation
- **Training pipeline**: Pretrain → Midtrain → SFT → DPO (optional)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all data collectors
python scripts/collect/collect_all.py

# Run specific collectors
python scripts/collect/collect_faa_handbooks.py
python scripts/collect/collect_ntsb.py
python scripts/collect/collect_metar.py
```

## Project Structure

```
FlightMind/
├── config.yaml              # Project configuration
├── requirements.txt         # Python dependencies
├── data/
│   ├── raw/                 # Downloaded source data (gitignored)
│   │   ├── faa_handbooks/
│   │   ├── faa_regulations/
│   │   ├── ntsb/
│   │   ├── asrs/
│   │   ├── metar/
│   │   ├── atc_transcripts/
│   │   ├── wikipedia_aviation/
│   │   ├── aircraft_performance/
│   │   └── ...
│   ├── cleaned/             # Processed text (gitignored)
│   ├── tokenized/           # Binary token shards (gitignored)
│   └── splits/              # Train/val/test splits
├── scripts/
│   ├── collect/             # Data collection scripts
│   │   ├── collect_all.py
│   │   ├── collect_faa_handbooks.py
│   │   ├── collect_faa_regulations.py
│   │   ├── collect_ntsb.py
│   │   ├── collect_asrs.py
│   │   ├── collect_metar.py
│   │   ├── collect_hf_datasets.py
│   │   ├── collect_wikipedia.py
│   │   └── collect_aircraft_performance.py
│   ├── process/             # Data cleaning and processing
│   └── utils/               # Shared utilities
├── tokenizer/               # Trained tokenizer files
├── model/                   # Model architecture code
├── train/                   # Training scripts
├── eval/                    # Evaluation and benchmarks
├── inference/               # Inference and demo UI
└── docs/                    # Step-by-step documentation
```

## Hardware

- **Data processing**: Dell 7920 (2x Xeon Gold 5118, 24c/48t, 64GB DDR4, RTX 4060)
- **Training**: Cloud 8x H100 SXM 80GB (RunPod / Vast.ai)
- **Inference/demo**: MacBook Pro M3 Pro (36GB unified memory)

## Cost Estimate

| Phase | Where | Est. Cost |
|-------|-------|-----------|
| Data collection + processing | Local | $0 |
| Pretraining (d20, ~50B tokens) | Cloud 8xH100 | $60-200 |
| Midtraining + SFT | Cloud | $20-60 |
| Evaluation + inference | Local | $0 |
| **Total** | | **$80-260** |

## License

TBD

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — nanochat, nanoGPT, llm.c
- [HuggingFace](https://huggingface.co) — FineWeb-EDU, datasets library
- FAA, NTSB, NASA — public aviation data
- OpenAP (TU Delft) — aircraft performance models

# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/shayn1825/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/shayn1825/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI PS5 trophy's Exophase earn rate at 2026-11-26 as five quantiles, from the house's recorded reads of the trophy page. A counts-based forecaster, written in Python 3.13 standard library only: it treats the page as a sequence of distinct states (Exophase refreshes its counts hours apart, frozen in between), fits the tracked-player count as a power law over the last 24 hours of states, and splits each trophy's jump in earners across refreshes into dilution (newly tracked players arriving with few trophies) versus progression (tracked players earning trophies over time), with the new cohort's attainment fixed by the launch-night refresh prior and the progression rate fitted per trophy, shrunk toward its tier. When the history holds no refresh at all, it falls back to a two-branch mixture over the next refresh landing or not. Backtests on the practice cases beat the persistence baseline by ~52% under every reconstruction of the post-refresh answer, and on 14 simulated final cases it beats both house baselines in 13 of 14.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=1c6jcmp6&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

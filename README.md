# 🏎️ F1 Pit Stop Strategy Optimization

A data-driven approach to optimizing Formula 1 pit stop strategies using real race data.

> ⚠️ **This project is currently a work in progress.**

## Overview

This project explores how to find the **optimal pit stop window** during an F1 race by analyzing tire degradation, pit stop time loss, and rival strategies. The goal is to build a system that can determine the best lap to pit and which compound to switch to — ultimately beating a simulated rival.

## How It Works

### 1. Real Race Data
Race data is loaded from the **2023 Italian Grand Prix (Monza)** using the [FastF1](https://github.com/theOehrly/Fast-F1) library. This includes lap times, tire compounds, tire age, pit stop events, and sector times for all 20 drivers.

### 2. Pit Loss Calculation
Pit stop cost is estimated by comparing average race pace against average pit-in/out lap times:
- **Avg Race Pace:** ~87.0s
- **Avg Pit Lap Time:** ~107.2s
- **Estimated Pit Loss:** ~20.3s

### 3. Tire Degradation Model
A simple tire degradation function models time loss per lap based on compound and tire age, including a **cliff** — a sharp performance drop-off after a threshold:

| Compound | Deg Rate   | Cliff Starts | Extra Deg After Cliff |
|----------|------------|--------------|----------------------|
| Soft     | 0.15 s/lap | Lap 15       | +0.50 s/lap          |
| Medium   | 0.08 s/lap | Lap 25       | +0.40 s/lap          |
| Hard     | 0.05 s/lap | Lap 35       | +0.30 s/lap          |

### 4. Rival Bot
A `RivalBot` class simulates a rival driver with a fixed pit stop strategy (e.g., pit on lap 24, switch from medium to hard). The goal is to optimize *your* strategy to beat this rival.

## Tech Stack

- **Python 3.11**
- **FastF1** — F1 telemetry and race data
- **Pandas** — Data manipulation
- **NumPy** — Numerical operations
- **Seaborn / Matplotlib** — Visualization

## Roadmap

- [x] Load and explore real F1 race data
- [x] Calculate pit stop time loss
- [x] Build tire degradation model with cliff
- [x] Create a basic rival bot
- [ ] Build full race simulation loop
- [ ] Implement Phase 2: Environment Design (The "Gym")
  - Use Gymnasium library to structure the RL environment
  - State Space: Current Lap, Current Tire Age, Current Tire Compound, Gap to Rival (normalized Box space)
  - Action Space: 0 (Stay out), 1 (Pit for Softs), 2 (Pit for Mediums), 3 (Pit for Hards)
  - Reward Function: Time-based rewards mimicking race time
- [ ] Implement strategy optimizer (find best pit lap)
- [ ] Visualize strategy outcomes
- [ ] Support multi-stop strategies
- [ ] Test across different circuits and seasons

## License

This project is for educational and personal use.

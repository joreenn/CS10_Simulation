# Hospital ER Simulation

A discrete-event simulation of a Hospital Emergency Room built with Python, SimPy, and Flask.

## Features

- **Discrete-Event Simulation**: Models patient flow through ER stages (Triage → Registration → Bed Assignment → Doctor Evaluation → Treatment → Discharge)
- **Web Interface**: Real-time visualization with animated patient flow, live charts, and KPIs
- **Statistical Analysis**: 30 replications with 95% confidence intervals
- **Resource Tracking**: Monitors utilization of Triage Nurses, Admin Staff, ER Beds, and Doctors

## Patient Flow

1. **Arrival** → Emergency entrance
2. **Triage** → Initial assessment (5-10 min, triangular distribution)
3. **Registration** → Admin paperwork (3-5 min, uniform distribution)
4. **Bed Assignment** → Wait for available ER bed
5. **Doctor Evaluation** → Initial diagnosis (10-30 min, triangular distribution)
6. **Treatment** → Nurses administer care (20-60 min, exponential distribution)
7. **Discharge** → Patient exits

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install simpy numpy pandas scipy flask flask-socketio
```

## Usage

### Terminal-Based Simulation

```bash
python er_simulation.py
```

Runs 30 replications and outputs comprehensive statistics including:
- Throughput and Length of Stay (LOS)
- Resource utilization
- Queue statistics
- 95% confidence intervals

### Web-Based Simulation

```bash
python app.py
```

Then open http://localhost:5000 in your browser.

#### Web Interface Features:

**Simulation Tab:**
- Configurable parameters (arrival rate, resource capacity, service times)
- Animated patient flow visualization
- Live KPI cards (arrivals, discharged, in system, bed occupancy)
- Real-time resource utilization bars
- Live charts (queue lengths, bed occupancy, throughput)

**Results Tab:**
- Summary statistics with confidence intervals
- Resource and queue performance summaries
- Comparison charts

**Details Tab:**
- Individual replication data
- Summary statistics table with CIs

## System Parameters

- **Arrival Rate**: 1 patient every 6 minutes (Exponential distribution)
- **Triage Nurse**: 1 nurse, 5-10 min service (ρ ≈ 1.25)
- **Admin Staff**: 1 staff, 3-5 min service (ρ ≈ 0.52)
- **ER Beds**: 10 beds
- **Doctors**: 2 doctors, 10-30 min service (ρ ≈ 1.88)
- **Nurses (Treatment)**: 4 nurses, 20-60 min service (ρ ≈ 1.0)
- **Simulation Duration**: 8,640 minutes (6 days)
- **Replications**: 30

## Technologies

- **Python 3.13+**
- **SimPy** - Discrete-event simulation framework
- **Flask** - Web framework
- **Flask-SocketIO** - Real-time WebSocket communication
- **Chart.js** - Client-side charting
- **NumPy, Pandas, SciPy** - Statistical analysis

## License

MIT License

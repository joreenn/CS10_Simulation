"""
=============================================================================
  Discrete-Event Simulation: Hospital Emergency Room
  Using Python + SimPy
=============================================================================
  Patient Flow:
    Arrival -> Triage -> Registration -> Bed Assignment -> Doctor Evaluation
    -> Treatment & Observation -> Discharge -> Release Bed -> Exit

  Resources:
    Triage Nurse (1), Admin Staff (1), ER Beds (10), Doctors (2), Nurses (4)

  Simulation: 30 independent replications, each 24 hours (1440 minutes).
=============================================================================
  NOTE ON WARM-UP:
    A warm-up period is useful only when the system can reach a steady state
    (all resource utilizations rho < 1).  With the given parameters the
    triage nurse is over-saturated (rho = 1.25) and the doctors are
    over-saturated (rho approx 1.88), so queues grow without bound and no
    steady state exists.  The warm-up is therefore set to 0 by default.
    Change WARMUP_PERIOD below if you adjust capacities or service times
    to create a stable system (e.g. 2 triage nurses, 4 doctors).
=============================================================================
"""

import simpy
import random
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

# =============================================================================
# SIMULATION PARAMETERS  (edit these to explore different scenarios)
# =============================================================================

SIM_DURATION        = 1440    # 24 hours of data collection (minutes)
WARMUP_PERIOD       = 0       # warm-up in minutes (set 7200 for 5-day warm-up
                              # if the system has rho < 1 for all resources)
TOTAL_SIM_TIME      = WARMUP_PERIOD + SIM_DURATION
NUM_REPLICATIONS    = 30      # independent replications
BED_SAMPLE_INTERVAL = 1       # sample bed occupancy every 1 minute

# Resource capacities
NUM_TRIAGE_NURSES   = 1
NUM_ADMIN_STAFF     = 1
NUM_ER_BEDS         = 10
NUM_DOCTORS         = 2
NUM_NURSES          = 4       # implied in delays, not explicitly seized

# Arrival distribution
MEAN_INTERARRIVAL   = 6       # Exponential mean (minutes) => ~10 patients/hour


# =============================================================================
# SERVICE-TIME DISTRIBUTION HELPERS
# =============================================================================

def arrival_interval(rng: random.Random) -> float:
    """Exponential inter-arrival time with mean = 6 minutes."""
    return rng.expovariate(1.0 / MEAN_INTERARRIVAL)


def triage_time(rng: random.Random) -> float:
    """Triangular(min=5, mode=7.5, max=10) minutes."""
    return rng.triangular(5, 10, 7.5)          # low, high, mode


def registration_time(rng: random.Random) -> float:
    """Uniform(3, 5) minutes."""
    return rng.uniform(3, 5)


def bed_assignment_time() -> float:
    """Nearly instant bed assignment."""
    return 0.01


def doctor_eval_time(rng: random.Random) -> float:
    """Triangular(min=15, mode=22.5, max=30) minutes."""
    return rng.triangular(15, 30, 22.5)


def treatment_observation_time(rng: random.Random) -> float:
    """Uniform(20, 60) minutes."""
    return rng.uniform(20, 60)


def discharge_time(rng: random.Random) -> float:
    """Triangular(min=5, mode=7.5, max=10) minutes."""
    return rng.triangular(5, 10, 7.5)


# =============================================================================
# DATA COLLECTION STRUCTURES
# =============================================================================

@dataclass
class PatientRecord:
    """Stores timestamps for a single patient traversal."""
    patient_id: int
    arrival_time: float       = 0.0
    triage_wait_start: float  = 0.0
    triage_start: float       = 0.0
    triage_end: float         = 0.0
    reg_wait_start: float     = 0.0
    reg_start: float          = 0.0
    reg_end: float            = 0.0
    bed_wait_start: float     = 0.0
    bed_assigned: float       = 0.0
    doctor_wait_start: float  = 0.0
    doctor_start: float       = 0.0
    doctor_end: float         = 0.0
    treatment_start: float    = 0.0
    treatment_end: float      = 0.0
    discharge_start: float    = 0.0
    discharge_end: float      = 0.0
    bed_released: float       = 0.0
    exit_time: float          = 0.0

    @property
    def wait_for_doctor(self) -> float:
        """Time from arrival until patient first sees a doctor."""
        return self.doctor_start - self.arrival_time

    @property
    def length_of_stay(self) -> float:
        """Total time in ER from arrival to exit."""
        return self.exit_time - self.arrival_time

    @property
    def triage_queue_wait(self) -> float:
        return self.triage_start - self.triage_wait_start

    @property
    def reg_queue_wait(self) -> float:
        return self.reg_start - self.reg_wait_start

    @property
    def bed_queue_wait(self) -> float:
        return self.bed_assigned - self.bed_wait_start

    @property
    def doctor_queue_wait(self) -> float:
        return self.doctor_start - self.doctor_wait_start


@dataclass
class ReplicationStats:
    """Aggregated statistics for one replication."""
    replication_id: int
    avg_wait_for_doctor: float    = 0.0
    avg_length_of_stay: float     = 0.0
    bed_occupancy_rate: float     = 0.0
    doctor_utilization: float     = 0.0
    triage_utilization: float     = 0.0
    admin_utilization: float      = 0.0
    daily_throughput: int         = 0
    total_arrivals: int           = 0
    max_triage_queue: int         = 0
    max_reg_queue: int            = 0
    max_bed_queue: int            = 0
    max_doctor_queue: int         = 0
    avg_triage_queue_wait: float  = 0.0
    avg_reg_queue_wait: float     = 0.0
    avg_bed_queue_wait: float     = 0.0
    avg_doctor_queue_wait: float  = 0.0


# =============================================================================
# SIMULATION MODEL
# =============================================================================

class ERSimulation:
    """Encapsulates one replication of the ER simulation."""

    def __init__(self, replication_id: int, seed: int):
        self.replication_id = replication_id
        self.rng = random.Random(seed)
        self.env = simpy.Environment()

        # ── Resources ──
        self.triage_nurse = simpy.Resource(self.env, capacity=NUM_TRIAGE_NURSES)
        self.admin_staff  = simpy.Resource(self.env, capacity=NUM_ADMIN_STAFF)
        self.er_bed       = simpy.Resource(self.env, capacity=NUM_ER_BEDS)
        self.doctor       = simpy.Resource(self.env, capacity=NUM_DOCTORS)

        # ── Data collection ──
        self.patient_records: List[PatientRecord] = []
        self.patient_counter   = 0
        self.arrivals_in_window = 0   # patients arriving during collection

        # Utilization tracking (busy-time accumulators for collection window)
        self.doctor_busy_time  = 0.0
        self.triage_busy_time  = 0.0
        self.admin_busy_time   = 0.0

        # Bed occupancy sampling
        self.bed_samples: List[float] = []

        # Queue-length tracking (peak during collection window)
        self.max_triage_queue = 0
        self.max_reg_queue    = 0
        self.max_bed_queue    = 0
        self.max_doctor_queue = 0

    # ── helper: accumulate busy time inside the collection window ──
    @staticmethod
    def _clamp_busy(start: float, end: float) -> float:
        """Return the portion of [start, end] that overlaps with the
        collection window [WARMUP_PERIOD, TOTAL_SIM_TIME]."""
        s = max(start, WARMUP_PERIOD)
        e = min(end, TOTAL_SIM_TIME)
        return max(0.0, e - s)

    def _in_window(self) -> bool:
        """True when the simulation clock is inside the collection window."""
        return self.env.now >= WARMUP_PERIOD

    # ─────────────────────────────────────────────
    # Patient process (the full ER journey)
    # ─────────────────────────────────────────────
    def patient_process(self, patient_id: int):
        rec = PatientRecord(patient_id=patient_id)
        rec.arrival_time = self.env.now

        if self._in_window():
            self.arrivals_in_window += 1

        # ── 1. TRIAGE ──
        rec.triage_wait_start = self.env.now
        with self.triage_nurse.request() as req:
            if self._in_window():
                q = len(self.triage_nurse.queue)
                if q > self.max_triage_queue:
                    self.max_triage_queue = q
            yield req
            rec.triage_start = self.env.now
            service = triage_time(self.rng)
            yield self.env.timeout(service)
            rec.triage_end = self.env.now
            self.triage_busy_time += self._clamp_busy(rec.triage_start,
                                                      rec.triage_end)

        # ── 2. REGISTRATION ──
        rec.reg_wait_start = self.env.now
        with self.admin_staff.request() as req:
            if self._in_window():
                q = len(self.admin_staff.queue)
                if q > self.max_reg_queue:
                    self.max_reg_queue = q
            yield req
            rec.reg_start = self.env.now
            service = registration_time(self.rng)
            yield self.env.timeout(service)
            rec.reg_end = self.env.now
            self.admin_busy_time += self._clamp_busy(rec.reg_start,
                                                     rec.reg_end)

        # ── 3. BED ASSIGNMENT (bed held until after discharge) ──
        rec.bed_wait_start = self.env.now
        bed_req = self.er_bed.request()
        if self._in_window():
            q = len(self.er_bed.queue)
            if q > self.max_bed_queue:
                self.max_bed_queue = q
        yield bed_req                       # SEIZE bed
        rec.bed_assigned = self.env.now
        yield self.env.timeout(bed_assignment_time())

        # ── 4. DOCTOR EVALUATION ──
        rec.doctor_wait_start = self.env.now
        with self.doctor.request() as req:
            if self._in_window():
                q = len(self.doctor.queue)
                if q > self.max_doctor_queue:
                    self.max_doctor_queue = q
            yield req
            rec.doctor_start = self.env.now
            service = doctor_eval_time(self.rng)
            yield self.env.timeout(service)
            rec.doctor_end = self.env.now
            self.doctor_busy_time += self._clamp_busy(rec.doctor_start,
                                                      rec.doctor_end)

        # ── 5. TREATMENT & OBSERVATION (delay only, bed still held) ──
        rec.treatment_start = self.env.now
        service = treatment_observation_time(self.rng)
        yield self.env.timeout(service)
        rec.treatment_end = self.env.now

        # ── 6. DISCHARGE (delay only, bed still held) ──
        rec.discharge_start = self.env.now
        service = discharge_time(self.rng)
        yield self.env.timeout(service)
        rec.discharge_end = self.env.now

        # ── 7. RELEASE BED ──
        self.er_bed.release(bed_req)        # RELEASE bed
        rec.bed_released = self.env.now

        # ── 8. EXIT ──
        rec.exit_time = self.env.now

        # Record patients whose exit falls inside the collection window.
        if rec.exit_time >= WARMUP_PERIOD and rec.exit_time <= TOTAL_SIM_TIME:
            self.patient_records.append(rec)

    # ─────────────────────────────────────────────
    # Patient arrival generator
    # ─────────────────────────────────────────────
    def patient_generator(self):
        """Generates patient arrivals for the full simulation duration."""
        while True:
            iat = arrival_interval(self.rng)
            yield self.env.timeout(iat)
            self.patient_counter += 1
            self.env.process(
                self.patient_process(self.patient_counter)
            )

    # ─────────────────────────────────────────────
    # Bed-occupancy sampler
    # ─────────────────────────────────────────────
    def bed_occupancy_sampler(self):
        """Samples bed utilization every BED_SAMPLE_INTERVAL minutes."""
        if self.env.now < WARMUP_PERIOD:
            yield self.env.timeout(WARMUP_PERIOD - self.env.now)
        while self.env.now < TOTAL_SIM_TIME:
            in_use = self.er_bed.count
            self.bed_samples.append(in_use / NUM_ER_BEDS)
            yield self.env.timeout(BED_SAMPLE_INTERVAL)

    # ─────────────────────────────────────────────
    # Queue-length sampler (periodic snapshot of all queues)
    # ─────────────────────────────────────────────
    def queue_length_sampler(self):
        """Periodically update max queue lengths during the collection window."""
        if self.env.now < WARMUP_PERIOD:
            yield self.env.timeout(WARMUP_PERIOD - self.env.now)
        while self.env.now < TOTAL_SIM_TIME:
            q_t = len(self.triage_nurse.queue)
            q_r = len(self.admin_staff.queue)
            q_b = len(self.er_bed.queue)
            q_d = len(self.doctor.queue)
            if q_t > self.max_triage_queue:
                self.max_triage_queue = q_t
            if q_r > self.max_reg_queue:
                self.max_reg_queue = q_r
            if q_b > self.max_bed_queue:
                self.max_bed_queue = q_b
            if q_d > self.max_doctor_queue:
                self.max_doctor_queue = q_d
            yield self.env.timeout(BED_SAMPLE_INTERVAL)

    # ─────────────────────────────────────────────
    # Run one replication
    # ─────────────────────────────────────────────
    def run(self) -> ReplicationStats:
        self.env.process(self.patient_generator())
        self.env.process(self.bed_occupancy_sampler())
        self.env.process(self.queue_length_sampler())
        self.env.run(until=TOTAL_SIM_TIME)

        stats = ReplicationStats(replication_id=self.replication_id)
        stats.total_arrivals = self.arrivals_in_window

        records = self.patient_records
        if not records:
            return stats

        stats.daily_throughput = len(records)

        waits   = [r.wait_for_doctor    for r in records]
        stays   = [r.length_of_stay     for r in records]
        t_waits = [r.triage_queue_wait  for r in records]
        r_waits = [r.reg_queue_wait     for r in records]
        b_waits = [r.bed_queue_wait     for r in records]
        d_waits = [r.doctor_queue_wait  for r in records]

        stats.avg_wait_for_doctor   = float(np.mean(waits))
        stats.avg_length_of_stay    = float(np.mean(stays))
        stats.avg_triage_queue_wait = float(np.mean(t_waits))
        stats.avg_reg_queue_wait    = float(np.mean(r_waits))
        stats.avg_bed_queue_wait    = float(np.mean(b_waits))
        stats.avg_doctor_queue_wait = float(np.mean(d_waits))

        # Bed occupancy (average of periodic samples)
        if self.bed_samples:
            stats.bed_occupancy_rate = float(np.mean(self.bed_samples)) * 100.0

        # Resource utilization = busy_time / (window * capacity)
        window = SIM_DURATION
        stats.doctor_utilization = (
            self.doctor_busy_time / (window * NUM_DOCTORS) * 100.0
        )
        stats.triage_utilization = (
            self.triage_busy_time / (window * NUM_TRIAGE_NURSES) * 100.0
        )
        stats.admin_utilization = (
            self.admin_busy_time / (window * NUM_ADMIN_STAFF) * 100.0
        )

        stats.max_triage_queue = self.max_triage_queue
        stats.max_reg_queue    = self.max_reg_queue
        stats.max_bed_queue    = self.max_bed_queue
        stats.max_doctor_queue = self.max_doctor_queue

        return stats


# =============================================================================
# MULTI-REPLICATION RUNNER & REPORTING
# =============================================================================

def run_all_replications() -> List[ReplicationStats]:
    """Run NUM_REPLICATIONS independent replications and return results."""
    results: List[ReplicationStats] = []
    base_seed = 42

    print("=" * 72)
    print("  Hospital Emergency Room - Discrete-Event Simulation (SimPy)")
    print("=" * 72)
    print(f"  Replications : {NUM_REPLICATIONS}")
    if WARMUP_PERIOD > 0:
        print(f"  Warm-up      : {WARMUP_PERIOD} min "
              f"({WARMUP_PERIOD / 1440:.1f} days)")
    else:
        print(f"  Warm-up      : None (fresh start)")
    print(f"  Run length   : {SIM_DURATION} min ({SIM_DURATION / 60:.0f} hours)")
    print(f"  Resources    : Triage Nurses={NUM_TRIAGE_NURSES}, "
          f"Admin Staff={NUM_ADMIN_STAFF}, ER Beds={NUM_ER_BEDS}, "
          f"Doctors={NUM_DOCTORS}")
    print("=" * 72)
    print()

    for rep in range(1, NUM_REPLICATIONS + 1):
        seed = base_seed + rep * 137          # distinct seed per replication
        sim = ERSimulation(replication_id=rep, seed=seed)
        stats = sim.run()
        results.append(stats)
        print(f"  Replication {rep:>2}/{NUM_REPLICATIONS} complete  "
              f"Throughput: {stats.daily_throughput:>4}  "
              f"Avg LOS: {stats.avg_length_of_stay:>7.1f} min  "
              f"Bed Occ: {stats.bed_occupancy_rate:>5.1f}%  "
              f"Doc Util: {stats.doctor_utilization:>5.1f}%")

    return results


def confidence_interval_95(data: np.ndarray):
    """Return (mean, lower, upper) for a 95% CI using the t-distribution."""
    from scipy import stats as sp_stats
    n = len(data)
    if n < 2:
        m = float(np.mean(data))
        return m, m, m
    mean = float(np.mean(data))
    se   = float(np.std(data, ddof=1)) / math.sqrt(n)
    t_crit = sp_stats.t.ppf(0.975, df=n - 1)
    return mean, mean - t_crit * se, mean + t_crit * se


def print_report(results: List[ReplicationStats]):
    """Print a comprehensive statistical report."""

    # Build a DataFrame for easy aggregation
    rows = []
    for r in results:
        rows.append({
            "Rep":                  r.replication_id,
            "Throughput":           r.daily_throughput,
            "Total Arrivals":       r.total_arrivals,
            "Avg Wait for Doctor":  r.avg_wait_for_doctor,
            "Avg Length of Stay":   r.avg_length_of_stay,
            "Bed Occupancy (%)":    r.bed_occupancy_rate,
            "Doctor Util (%)":      r.doctor_utilization,
            "Triage Util (%)":      r.triage_utilization,
            "Admin Util (%)":       r.admin_utilization,
            "Max Triage Q":         r.max_triage_queue,
            "Max Reg Q":            r.max_reg_queue,
            "Max Bed Q":            r.max_bed_queue,
            "Max Doctor Q":         r.max_doctor_queue,
            "Avg Triage Q Wait":    r.avg_triage_queue_wait,
            "Avg Reg Q Wait":       r.avg_reg_queue_wait,
            "Avg Bed Q Wait":       r.avg_bed_queue_wait,
            "Avg Doctor Q Wait":    r.avg_doctor_queue_wait,
        })
    df = pd.DataFrame(rows)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 220)
    pd.set_option('display.float_format', '{:.2f}'.format)

    # ── Section 1: Individual Replication Results ──
    print("\n")
    print("=" * 120)
    print("  SECTION 1: INDIVIDUAL REPLICATION RESULTS")
    print("=" * 120)

    display_cols = [
        "Rep", "Throughput", "Total Arrivals",
        "Avg Wait for Doctor", "Avg Length of Stay",
        "Bed Occupancy (%)", "Doctor Util (%)",
        "Triage Util (%)", "Admin Util (%)",
    ]
    print(df[display_cols].to_string(index=False))

    print()
    queue_cols = [
        "Rep",
        "Avg Triage Q Wait", "Max Triage Q",
        "Avg Reg Q Wait", "Max Reg Q",
        "Avg Bed Q Wait", "Max Bed Q",
        "Avg Doctor Q Wait", "Max Doctor Q",
    ]
    print(df[queue_cols].to_string(index=False))

    # ── Section 2: Summary Statistics ──
    metric_cols = [c for c in df.columns if c != "Rep"]
    print("\n")
    print("=" * 120)
    print("  SECTION 2: SUMMARY STATISTICS ACROSS ALL REPLICATIONS")
    print("=" * 120)

    summary_rows = []
    for col in metric_cols:
        data = df[col].values.astype(float)
        mean_val = float(np.mean(data))
        std_val  = float(np.std(data, ddof=1))
        min_val  = float(np.min(data))
        max_val  = float(np.max(data))
        _, ci_lo, ci_hi = confidence_interval_95(data)
        summary_rows.append({
            "Metric":      col,
            "Mean":        mean_val,
            "Std Dev":     std_val,
            "Min":         min_val,
            "Max":         max_val,
            "95% CI Low":  ci_lo,
            "95% CI High": ci_hi,
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # ── Section 3: Resource Utilization ──
    print("\n")
    print("=" * 120)
    print("  SECTION 3: RESOURCE UTILIZATION SUMMARY")
    print("=" * 120)

    resources = [
        ("Triage Nurse",    NUM_TRIAGE_NURSES, "Triage Util (%)"),
        ("Admin Staff",     NUM_ADMIN_STAFF,   "Admin Util (%)"),
        ("ER Beds",         NUM_ER_BEDS,       "Bed Occupancy (%)"),
        ("Doctors",         NUM_DOCTORS,       "Doctor Util (%)"),
        ("Nurses (implied)",NUM_NURSES,        None),
    ]

    print(f"  {'Resource':<20} {'Capacity':>8} {'Mean Util %':>12} "
          f"{'Std Dev':>10} {'95% CI':>24}")
    print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*10} {'-'*24}")

    for name, cap, col in resources:
        if col is None:
            print(f"  {name:<20} {cap:>8} {'N/A (implied)':>12}")
            continue
        data = df[col].values.astype(float)
        m, lo, hi = confidence_interval_95(data)
        sd = float(np.std(data, ddof=1))
        print(f"  {name:<20} {cap:>8} {m:>11.2f}% {sd:>10.2f} "
              f"[{lo:>8.2f}%, {hi:>8.2f}%]")

    # ── Section 4: Queue Statistics ──
    print("\n")
    print("=" * 120)
    print("  SECTION 4: QUEUE STATISTICS SUMMARY")
    print("=" * 120)

    queue_metrics = [
        ("Triage",        "Avg Triage Q Wait", "Max Triage Q"),
        ("Registration",  "Avg Reg Q Wait",    "Max Reg Q"),
        ("Bed Assignment","Avg Bed Q Wait",    "Max Bed Q"),
        ("Doctor",        "Avg Doctor Q Wait", "Max Doctor Q"),
    ]

    print(f"  {'Queue':<18} {'Avg Wait (min)':>16} {'95% CI Wait':>26} "
          f"{'Avg Max Q Len':>14} {'Max Q (range)':>18}")
    print(f"  {'-'*18} {'-'*16} {'-'*26} {'-'*14} {'-'*18}")
    for label, wait_col, max_col in queue_metrics:
        w_data = df[wait_col].values.astype(float)
        m_data = df[max_col].values.astype(float)
        w_mean, w_lo, w_hi = confidence_interval_95(w_data)
        m_mean = float(np.mean(m_data))
        m_min  = int(np.min(m_data))
        m_max  = int(np.max(m_data))
        print(f"  {label:<18} {w_mean:>15.2f}  [{w_lo:>10.2f}, {w_hi:>10.2f}] "
              f"{m_mean:>13.1f}   [{m_min:>5} - {m_max:>5}]")

    # ── Section 5: Key Performance Indicators ──
    print("\n")
    print("=" * 120)
    print("  SECTION 5: KEY PERFORMANCE INDICATORS (KPIs)")
    print("=" * 120)

    kpis = [
        ("Avg Wait for Doctor (min)", "Avg Wait for Doctor"),
        ("Avg Length of Stay  (min)", "Avg Length of Stay"),
        ("Bed Occupancy Rate   (%)",  "Bed Occupancy (%)"),
        ("Doctor Utilization   (%)",  "Doctor Util (%)"),
        ("Triage Nurse Util    (%)",  "Triage Util (%)"),
        ("Admin Staff Util     (%)",  "Admin Util (%)"),
        ("Daily Throughput    (pts)", "Throughput"),
    ]

    for label, col in kpis:
        data = df[col].values.astype(float)
        m, lo, hi = confidence_interval_95(data)
        if "min" in label:
            extra = f"  ({m/60:.1f} hrs)" if m > 60 else ""
            print(f"  {label} : {m:>8.2f}{extra}   "
                  f"95% CI [{lo:.2f}, {hi:.2f}]")
        elif "pts" in label:
            print(f"  {label} : {m:>8.1f}          "
                  f"95% CI [{lo:.1f}, {hi:.1f}]")
        else:
            print(f"  {label} : {m:>8.2f}          "
                  f"95% CI [{lo:.2f}, {hi:.2f}]")

    print("\n" + "=" * 120)
    print("  Simulation complete.")
    print("=" * 120)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = run_all_replications()
    print_report(results)

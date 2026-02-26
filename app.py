"""
=============================================================================
  Hospital ER Simulation — Web Interface (Flask + Socket.IO)
=============================================================================
  Runs the same SimPy discrete-event simulation but streams live updates
  to a browser dashboard via WebSockets.
=============================================================================
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import simpy, random, math, threading, time
import numpy as np
from dataclasses import dataclass, asdict
from typing import List

app = Flask(__name__)
app.config["SECRET_KEY"] = "er-sim-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ─── Global state ───
simulation_thread = None
sim_running = False

# =============================================================================
# SIMULATION PARAMETERS (defaults — overridden by UI)
# =============================================================================
DEFAULT_PARAMS = {
    "sim_duration": 1440,
    "warmup_period": 0,
    "num_replications": 30,
    "mean_interarrival": 6,
    "num_triage_nurses": 1,
    "num_admin_staff": 1,
    "num_er_beds": 10,
    "num_doctors": 2,
    "num_nurses": 4,
}

# =============================================================================
# SERVICE-TIME HELPERS
# =============================================================================
def arrival_interval(rng, mean):
    return rng.expovariate(1.0 / mean)

def triage_time(rng):
    return rng.triangular(5, 10, 7.5)

def registration_time(rng):
    return rng.uniform(3, 5)

def doctor_eval_time(rng):
    return rng.triangular(15, 30, 22.5)

def treatment_observation_time(rng):
    return rng.uniform(20, 60)

def discharge_time(rng):
    return rng.triangular(5, 10, 7.5)

# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class PatientRecord:
    patient_id: int
    arrival_time: float = 0.0
    triage_wait_start: float = 0.0
    triage_start: float = 0.0
    triage_end: float = 0.0
    reg_wait_start: float = 0.0
    reg_start: float = 0.0
    reg_end: float = 0.0
    bed_wait_start: float = 0.0
    bed_assigned: float = 0.0
    doctor_wait_start: float = 0.0
    doctor_start: float = 0.0
    doctor_end: float = 0.0
    treatment_start: float = 0.0
    treatment_end: float = 0.0
    discharge_start: float = 0.0
    discharge_end: float = 0.0
    exit_time: float = 0.0

# =============================================================================
# SIMULATION ENGINE
# =============================================================================
class ERSimulation:
    def __init__(self, rep_id, seed, params, socketio_ref, emit_live=False):
        self.rep_id = rep_id
        self.rng = random.Random(seed)
        self.env = simpy.Environment()
        self.p = params
        self.sio = socketio_ref
        self.emit_live = emit_live

        self.triage_nurse = simpy.Resource(self.env, capacity=params["num_triage_nurses"])
        self.admin_staff = simpy.Resource(self.env, capacity=params["num_admin_staff"])
        self.er_bed = simpy.Resource(self.env, capacity=params["num_er_beds"])
        self.doctor = simpy.Resource(self.env, capacity=params["num_doctors"])

        self.records: List[PatientRecord] = []
        self.patient_counter = 0
        self.arrivals = 0
        self.discharged = 0
        self.doctor_busy = 0.0
        self.triage_busy = 0.0
        self.admin_busy = 0.0
        self.bed_samples = []
        self.max_q = {"triage": 0, "registration": 0, "bed": 0, "doctor": 0}

        # Patient location tracking (live counts)
        self.patients_at = {
            "triage": 0,
            "triage_queue": 0,
            "registration": 0,
            "registration_queue": 0,
            "bed_queue": 0,
            "in_bed": 0,
            "doctor": 0,
            "doctor_queue": 0,
            "treatment": 0,
            "discharge": 0,
        }

        # ── Arena-style entity tracking ──
        self.available_beds = list(range(1, params["num_er_beds"] + 1))
        self.patient_entities = {}   # pid -> {station, status, bed}
        self.staff_entities = {}     # staff_id -> {type, status, bed}
        for i in range(params["num_doctors"]):
            self.staff_entities[f"d{i}"] = {"type": "doctor", "status": "idle", "bed": None}
        for i in range(params["num_nurses"]):
            self.staff_entities[f"n{i}"] = {"type": "nurse", "status": "idle", "bed": None}

        # For live event log
        self.event_log = []
        self.last_emit_time = -1

        # Time-series data for charts
        self.ts_data = []

        self.wp = params["warmup_period"]
        self.total_time = params["warmup_period"] + params["sim_duration"]

    def _clamp(self, s, e):
        s2 = max(s, self.wp)
        e2 = min(e, self.total_time)
        return max(0.0, e2 - s2)

    def _in_window(self):
        return self.env.now >= self.wp

    def _log_event(self, patient_id, event, station):
        if self._in_window():
            self.event_log.append({
                "time": round(self.env.now - self.wp, 2),
                "patient": patient_id,
                "event": event,
                "station": station,
            })

    # ── Entity helpers ──
    def _take_bed(self):
        return self.available_beds.pop(0) if self.available_beds else 1

    def _free_bed(self, b):
        if b is not None:
            self.available_beds.append(b)
            self.available_beds.sort()

    def _take_staff(self, stype, bed):
        prefix = "d" if stype == "doctor" else "n"
        for sid, info in self.staff_entities.items():
            if sid.startswith(prefix) and info["status"] == "idle":
                info["status"] = "busy"
                info["bed"] = bed
                return sid
        return None

    def _free_staff(self, sid):
        if sid and sid in self.staff_entities:
            self.staff_entities[sid]["status"] = "idle"
            self.staff_entities[sid]["bed"] = None

    def _entities_snapshot(self):
        ents = []
        for pid, info in self.patient_entities.items():
            ents.append({"id": f"p{pid}", "tp": "P", "st": info["station"],
                         "ss": info["status"], "bed": info.get("bed")})
        for sid, info in self.staff_entities.items():
            ents.append({"id": sid, "tp": "D" if info["type"] == "doctor" else "N",
                         "ss": info["status"], "bed": info.get("bed")})
        return ents

    def patient_process(self, pid):
        global sim_running
        if not sim_running:
            return

        rec = PatientRecord(patient_id=pid)
        rec.arrival_time = self.env.now

        if self._in_window():
            self.arrivals += 1
            self._log_event(pid, "arrived", "entrance")

        # Entity tracking — add patient
        self.patient_entities[pid] = {"station": "entrance", "status": "arriving", "bed": None}

        # 1. TRIAGE
        rec.triage_wait_start = self.env.now
        self.patients_at["triage_queue"] += 1
        self.patient_entities[pid]["station"] = "triage_queue"
        self.patient_entities[pid]["status"] = "triage_queue"
        with self.triage_nurse.request() as req:
            if self._in_window():
                q = len(self.triage_nurse.queue)
                self.max_q["triage"] = max(self.max_q["triage"], q)
            yield req
            self.patients_at["triage_queue"] -= 1
            self.patients_at["triage"] += 1
            self.patient_entities[pid]["station"] = "triage"
            self.patient_entities[pid]["status"] = "triage"
            rec.triage_start = self.env.now
            self._log_event(pid, "start", "triage")
            svc = triage_time(self.rng)
            yield self.env.timeout(svc)
            rec.triage_end = self.env.now
            self.triage_busy += self._clamp(rec.triage_start, rec.triage_end)
            self.patients_at["triage"] -= 1
            self._log_event(pid, "done", "triage")

        # 2. REGISTRATION
        rec.reg_wait_start = self.env.now
        self.patients_at["registration_queue"] += 1
        self.patient_entities[pid]["station"] = "reg_queue"
        self.patient_entities[pid]["status"] = "reg_queue"
        with self.admin_staff.request() as req:
            if self._in_window():
                q = len(self.admin_staff.queue)
                self.max_q["registration"] = max(self.max_q["registration"], q)
            yield req
            self.patients_at["registration_queue"] -= 1
            self.patients_at["registration"] += 1
            self.patient_entities[pid]["station"] = "registration"
            self.patient_entities[pid]["status"] = "registration"
            rec.reg_start = self.env.now
            self._log_event(pid, "start", "registration")
            svc = registration_time(self.rng)
            yield self.env.timeout(svc)
            rec.reg_end = self.env.now
            self.admin_busy += self._clamp(rec.reg_start, rec.reg_end)
            self.patients_at["registration"] -= 1
            self._log_event(pid, "done", "registration")

        # 3. BED ASSIGNMENT
        rec.bed_wait_start = self.env.now
        self.patients_at["bed_queue"] += 1
        self.patient_entities[pid]["station"] = "bed_queue"
        self.patient_entities[pid]["status"] = "bed_queue"
        bed_req = self.er_bed.request()
        if self._in_window():
            q = len(self.er_bed.queue)
            self.max_q["bed"] = max(self.max_q["bed"], q)
        yield bed_req
        self.patients_at["bed_queue"] -= 1
        self.patients_at["in_bed"] += 1
        bed_num = self._take_bed()
        self.patient_entities[pid]["station"] = f"bed_{bed_num}"
        self.patient_entities[pid]["status"] = "in_bed"
        self.patient_entities[pid]["bed"] = bed_num
        rec.bed_assigned = self.env.now
        self._log_event(pid, "assigned", "bed")
        yield self.env.timeout(0.01)

        # 4. DOCTOR EVAL
        rec.doctor_wait_start = self.env.now
        self.patients_at["doctor_queue"] += 1
        self.patient_entities[pid]["status"] = "waiting_doctor"
        with self.doctor.request() as req:
            if self._in_window():
                q = len(self.doctor.queue)
                self.max_q["doctor"] = max(self.max_q["doctor"], q)
            yield req
            self.patients_at["doctor_queue"] -= 1
            self.patients_at["doctor"] += 1
            doc_id = self._take_staff("doctor", bed_num)
            self.patient_entities[pid]["status"] = "with_doctor"
            rec.doctor_start = self.env.now
            self._log_event(pid, "start", "doctor")
            svc = doctor_eval_time(self.rng)
            yield self.env.timeout(svc)
            rec.doctor_end = self.env.now
            self.doctor_busy += self._clamp(rec.doctor_start, rec.doctor_end)
            self._free_staff(doc_id)
            self.patients_at["doctor"] -= 1
            self._log_event(pid, "done", "doctor")

        # 5. TREATMENT
        rec.treatment_start = self.env.now
        self.patients_at["treatment"] += 1
        nurse_id = self._take_staff("nurse", bed_num)
        self.patient_entities[pid]["status"] = "treatment"
        self._log_event(pid, "start", "treatment")
        svc = treatment_observation_time(self.rng)
        yield self.env.timeout(svc)
        rec.treatment_end = self.env.now
        self._free_staff(nurse_id)
        self.patients_at["treatment"] -= 1
        self._log_event(pid, "done", "treatment")

        # 6. DISCHARGE
        rec.discharge_start = self.env.now
        self.patients_at["discharge"] += 1
        self.patient_entities[pid]["status"] = "discharge"
        self._log_event(pid, "start", "discharge")
        svc = discharge_time(self.rng)
        yield self.env.timeout(svc)
        rec.discharge_end = self.env.now
        self.patients_at["discharge"] -= 1
        self._log_event(pid, "done", "discharge")

        # 7. RELEASE BED
        self.patients_at["in_bed"] -= 1
        self._free_bed(bed_num)
        self.er_bed.release(bed_req)

        # 8. EXIT
        self.patient_entities[pid]["station"] = "exit"
        self.patient_entities[pid]["status"] = "exiting"
        rec.exit_time = self.env.now
        if rec.exit_time >= self.wp and rec.exit_time <= self.total_time:
            self.records.append(rec)
            self.discharged += 1
            self._log_event(pid, "exited", "exit")

        # Remove from entity tracker
        self.patient_entities.pop(pid, None)

    def patient_generator(self):
        global sim_running
        while sim_running:
            iat = arrival_interval(self.rng, self.p["mean_interarrival"])
            yield self.env.timeout(iat)
            if not sim_running:
                break
            self.patient_counter += 1
            self.env.process(self.patient_process(self.patient_counter))

    def sampler(self):
        if self.env.now < self.wp:
            yield self.env.timeout(self.wp - self.env.now)
        sample_interval = max(1, self.p["sim_duration"] // 200)
        while self.env.now < self.total_time:
            bed_occ = self.er_bed.count / self.p["num_er_beds"]
            self.bed_samples.append(bed_occ)
            t = round(self.env.now - self.wp, 2)
            self.ts_data.append({
                "time": t,
                "bed_occ": round(bed_occ * 100, 1),
                "q_triage": len(self.triage_nurse.queue),
                "q_reg": len(self.admin_staff.queue),
                "q_bed": len(self.er_bed.queue),
                "q_doctor": len(self.doctor.queue),
                "arrivals": self.arrivals,
                "discharged": self.discharged,
            })

            q_t = len(self.triage_nurse.queue)
            q_r = len(self.admin_staff.queue)
            q_b = len(self.er_bed.queue)
            q_d = len(self.doctor.queue)
            self.max_q["triage"] = max(self.max_q["triage"], q_t)
            self.max_q["registration"] = max(self.max_q["registration"], q_r)
            self.max_q["bed"] = max(self.max_q["bed"], q_b)
            self.max_q["doctor"] = max(self.max_q["doctor"], q_d)

            yield self.env.timeout(sample_interval)

    def live_emitter(self):
        """Emit snapshots to the browser during the live replication."""
        if self.env.now < self.wp:
            yield self.env.timeout(self.wp - self.env.now)
        emit_interval = max(1, self.p["sim_duration"] // 100)
        while self.env.now < self.total_time:
            if self.emit_live and sim_running:
                dur = self.p["sim_duration"]
                elapsed = self.env.now - self.wp
                snapshot = {
                    "sim_time": round(elapsed, 1),
                    "sim_hours": f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
                    "progress": round(elapsed / dur * 100, 1),
                    "arrivals": self.arrivals,
                    "discharged": self.discharged,
                    "in_system": self.arrivals - self.discharged,
                    "beds_used": self.er_bed.count,
                    "beds_total": self.p["num_er_beds"],
                    "bed_occ": round(self.er_bed.count / self.p["num_er_beds"] * 100, 1),
                    "q_triage": len(self.triage_nurse.queue),
                    "q_reg": len(self.admin_staff.queue),
                    "q_bed": len(self.er_bed.queue),
                    "q_doctor": len(self.doctor.queue),
                    "triage_busy": round(self.triage_busy / (max(0.001, elapsed) * self.p["num_triage_nurses"]) * 100, 1),
                    "admin_busy": round(self.admin_busy / (max(0.001, elapsed) * self.p["num_admin_staff"]) * 100, 1),
                    "doctor_busy": round(self.doctor_busy / (max(0.001, elapsed) * self.p["num_doctors"]) * 100, 1),
                    "patients_at": dict(self.patients_at),
                    "entities": self._entities_snapshot(),
                }
                self.sio.emit("live_snapshot", snapshot)
                self.sio.sleep(0)
            yield self.env.timeout(emit_interval)

    def run(self):
        self.env.process(self.patient_generator())
        self.env.process(self.sampler())
        if self.emit_live:
            self.env.process(self.live_emitter())
        self.env.run(until=self.total_time)
        return self._compute_stats()

    def _compute_stats(self):
        recs = self.records
        dur = self.p["sim_duration"]
        s = {
            "rep": self.rep_id,
            "throughput": len(recs),
            "total_arrivals": self.arrivals,
            "max_triage_q": self.max_q["triage"],
            "max_reg_q": self.max_q["registration"],
            "max_bed_q": self.max_q["bed"],
            "max_doctor_q": self.max_q["doctor"],
        }
        if not recs:
            for k in ["avg_wait_doctor", "avg_los", "bed_occ", "doctor_util",
                       "triage_util", "admin_util", "avg_triage_wait",
                       "avg_reg_wait", "avg_bed_wait", "avg_doctor_wait"]:
                s[k] = 0.0
            s["ts_data"] = self.ts_data
            return s

        waits = [r.doctor_start - r.arrival_time for r in recs]
        stays = [r.exit_time - r.arrival_time for r in recs]
        tw = [r.triage_start - r.triage_wait_start for r in recs]
        rw = [r.reg_start - r.reg_wait_start for r in recs]
        bw = [r.bed_assigned - r.bed_wait_start for r in recs]
        dw = [r.doctor_start - r.doctor_wait_start for r in recs]

        s["avg_wait_doctor"] = round(float(np.mean(waits)), 2)
        s["avg_los"] = round(float(np.mean(stays)), 2)
        s["avg_triage_wait"] = round(float(np.mean(tw)), 2)
        s["avg_reg_wait"] = round(float(np.mean(rw)), 2)
        s["avg_bed_wait"] = round(float(np.mean(bw)), 2)
        s["avg_doctor_wait"] = round(float(np.mean(dw)), 2)
        s["bed_occ"] = round(float(np.mean(self.bed_samples)) * 100, 2) if self.bed_samples else 0.0
        s["doctor_util"] = round(self.doctor_busy / (dur * self.p["num_doctors"]) * 100, 2)
        s["triage_util"] = round(self.triage_busy / (dur * self.p["num_triage_nurses"]) * 100, 2)
        s["admin_util"] = round(self.admin_busy / (dur * self.p["num_admin_staff"]) * 100, 2)
        s["ts_data"] = self.ts_data
        return s


# =============================================================================
# CONFIDENCE INTERVAL
# =============================================================================
def ci95(values):
    from scipy import stats as sp
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 2:
        m = float(np.mean(arr))
        return {"mean": m, "lo": m, "hi": m, "std": 0.0, "min": m, "max": m}
    m = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    se = sd / math.sqrt(n)
    t = sp.t.ppf(0.975, df=n - 1)
    return {
        "mean": round(m, 2),
        "lo": round(m - t * se, 2),
        "hi": round(m + t * se, 2),
        "std": round(sd, 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
    }


# =============================================================================
# SIMULATION RUNNER (background thread)
# =============================================================================
def run_simulation(params):
    global sim_running
    sim_running = True
    base_seed = 42
    num_reps = params["num_replications"]
    all_stats = []

    socketio.emit("sim_started", {"num_reps": num_reps, "params": params})

    for rep in range(1, num_reps + 1):
        if not sim_running:
            break

        seed = base_seed + rep * 137
        emit_live = (rep == 1)  # stream live view for first replication
        sim = ERSimulation(rep, seed, params, socketio, emit_live=emit_live)

        if emit_live:
            socketio.emit("live_rep_start", {"rep": rep})

        stats = sim.run()

        if not sim_running:
            break

        all_stats.append(stats)

        # Send replication result
        rep_msg = {
            "rep": rep,
            "total": num_reps,
            "stats": {k: v for k, v in stats.items() if k != "ts_data"},
            "progress": round(rep / num_reps * 100, 1),
        }
        if emit_live:
            rep_msg["ts_data"] = stats["ts_data"]
        socketio.emit("rep_complete", rep_msg)
        socketio.sleep(0)

    if sim_running and all_stats:
        # Compute summary across all replications
        keys = ["throughput", "total_arrivals", "avg_wait_doctor", "avg_los",
                "bed_occ", "doctor_util", "triage_util", "admin_util",
                "avg_triage_wait", "avg_reg_wait", "avg_bed_wait", "avg_doctor_wait",
                "max_triage_q", "max_reg_q", "max_bed_q", "max_doctor_q"]

        summary = {}
        for k in keys:
            vals = [s[k] for s in all_stats]
            summary[k] = ci95(vals)

        socketio.emit("sim_complete", {
            "summary": summary,
            "all_reps": [{k: v for k, v in s.items() if k != "ts_data"} for s in all_stats],
            "params": params,
        })
    elif not all_stats:
        socketio.emit("sim_stopped", {})

    sim_running = False


# =============================================================================
# ROUTES
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html", defaults=DEFAULT_PARAMS)


@socketio.on("start_simulation")
def handle_start(data):
    global simulation_thread, sim_running
    if sim_running:
        emit("error", {"msg": "Simulation already running."})
        return

    params = {
        "sim_duration": int(data.get("sim_duration", 1440)),
        "warmup_period": int(data.get("warmup_period", 0)),
        "num_replications": int(data.get("num_replications", 30)),
        "mean_interarrival": float(data.get("mean_interarrival", 6)),
        "num_triage_nurses": int(data.get("num_triage_nurses", 1)),
        "num_admin_staff": int(data.get("num_admin_staff", 1)),
        "num_er_beds": int(data.get("num_er_beds", 10)),
        "num_doctors": int(data.get("num_doctors", 2)),
        "num_nurses": int(data.get("num_nurses", 4)),
    }

    simulation_thread = threading.Thread(target=run_simulation, args=(params,), daemon=True)
    simulation_thread.start()


@socketio.on("stop_simulation")
def handle_stop():
    global sim_running
    sim_running = False


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5000, debug=False, allow_unsafe_werkzeug=True)

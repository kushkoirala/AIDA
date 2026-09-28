#!/usr/bin/env python3
"""Smoke check for classical XC controller quality fixes.

Checks, without the telemetry server or the LLM path:

1. Turn-point distance scales with the direct route
   (SN65→KAAO short leg, SN65→K50K long leg, and a synthetic floor case).
2. TURN_TO_INTERCEPT uses the cruise energy hold instead of a fixed 0.7
   throttle and frozen cruise pitch.
3. A short closed-loop turn on the database Cessna does not reproduce the
   live-flight balloon (~+370 ft) and airspeed spike (~120→135 kt).
4. FlightSimulator for model ``cessna172`` is built from
   ``aircraft_database.get_aircraft``, and an unknown id falls back.
   The Cessna lateral set must be the trainer derivatives, not the Udaan
   Table 2 block (that paste departs in INITIAL_CLIMB).
5. Database-backed Cessna takeoff on SN65 climbs out of INITIAL_CLIMB
   into CLIMB without the ~170 ft / 85 kt energy collapse.

Re-run the live server the way PackScale does::

    python scripts/run_dynamic_xc.py --speed 3

Then send ``start_flight`` with origin SN65 and destination KAAO (viewer
Fly button, or a websocket client on port 8765). The banner prints the
scaled turn point and ``Dynamics: cessna172 via aircraft_database``.
Overrides (``set_altitude``, ``set_heading``, ``land``) are unchanged.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "gpu-flight-dynamics" / "python"))

from aircraft_database import get_aircraft
from flight_dynamics import (
    AircraftParams, FlightSimulator, StateIndex, aircraft_model_from_command,
    aircraft_params_from_config, resolve_aircraft_params,
)
from generalized_xc_controller import (
    AirportConfig, GeneralizedXCController, KANSAS_AIRPORTS, NM_TO_FT,
    XCPhase, adaptive_turn_point_distance_nm, create_controller,
)

M_TO_FT = 3.28084
FT_TO_M = 0.3048
KTS_TO_MPS = 0.514444


def _check(name, ok, detail):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    if not ok:
        raise AssertionError(f"{name}: {detail}")


def _make_state(alt_ft, airspeed_kts, heading_deg, climb_fpm=0.0, roll_deg=0.0):
    state = np.zeros(12, dtype=np.float32)
    state[StateIndex.Z] = -(alt_ft * FT_TO_M)
    state[StateIndex.U] = airspeed_kts * KTS_TO_MPS
    # Controller climb rate is -w (body), positive up.
    w_fps = -(climb_fpm / 60.0)
    state[StateIndex.W] = w_fps * FT_TO_M
    state[StateIndex.PSI] = np.deg2rad(heading_deg)
    state[StateIndex.PHI] = np.deg2rad(roll_deg)
    return state


def test_turn_point_scaling():
    print("\nTurn-point scaling")
    kaao = create_controller("SN65", "KAAO", cruise_altitude_ft=5500.0)
    k50k = create_controller("SN65", "K50K", cruise_altitude_ft=5500.0)

    kaao_expected = adaptive_turn_point_distance_nm(kaao.direct_distance_nm)
    k50k_expected = adaptive_turn_point_distance_nm(k50k.direct_distance_nm)
    _check(
        "SN65-KAAO adaptive TP",
        abs(kaao.tp_distance_nm - kaao_expected) < 1e-6
        and 3.0 <= kaao.tp_distance_nm < 10.0,
        f"direct {kaao.direct_distance_nm:.2f} nm → TP {kaao.tp_distance_nm:.2f} nm "
        f"(expected {kaao_expected:.2f}, was fixed 10)",
    )
    _check(
        "SN65-K50K keeps 10 nm cap",
        abs(k50k.tp_distance_nm - 10.0) < 1e-6 and k50k.direct_distance_nm > 50.0,
        f"direct {k50k.direct_distance_nm:.2f} nm → TP {k50k.tp_distance_nm:.2f} nm",
    )

    # A stale explicit 10 nm request must not fly past the short leg.
    capped = create_controller("SN65", "KAAO", tp_distance_nm=10.0)
    _check(
        "explicit 10 nm capped on SN65-KAAO",
        abs(capped.tp_distance_nm - kaao.tp_distance_nm) < 1e-6,
        f"requested 10 → {capped.tp_distance_nm:.2f} nm",
    )

    # Shorter explicit request on a long leg is kept.
    shorter = create_controller("SN65", "K50K", tp_distance_nm=6.0)
    _check(
        "explicit 6 nm kept on SN65-K50K",
        abs(shorter.tp_distance_nm - 6.0) < 1e-6,
        f"TP {shorter.tp_distance_nm:.2f} nm",
    )

    # Floor: a 4 nm leg would be 1.6 nm at 0.4×, so the 3 nm floor applies.
    origin = AirportConfig("AAAA", "Short Origin", x_ft=0.0, y_ft=0.0,
                           runway_heading_deg=360.0)
    dest = AirportConfig("BBBB", "Short Dest", x_ft=4.0 * NM_TO_FT, y_ft=0.0,
                         runway_heading_deg=180.0)
    short = GeneralizedXCController(origin, dest, cruise_altitude_ft=3500.0)
    _check(
        "3 nm floor on a 4 nm leg",
        abs(short.direct_distance_nm - 4.0) < 0.05 and abs(short.tp_distance_nm - 3.0) < 1e-6,
        f"direct {short.direct_distance_nm:.2f} nm → TP {short.tp_distance_nm:.2f} nm",
    )


def test_turn_energy_hold_command():
    print("\nTURN_TO_INTERCEPT energy hold (open loop)")
    ctrl = create_controller("SN65", "KAAO", cruise_altitude_ft=5500.0)
    ctrl.phase = XCPhase.TURN_TO_INTERCEPT
    # Runway is 180°. Start well off that heading so this is a real turn.
    on_target = _make_state(5500.0, 110.0, heading_deg=40.0)
    action = ctrl.compute_action(on_target, 0.0)
    _check(
        "on-target throttle is cruise, not 0.7",
        0.45 <= float(action[0]) <= 0.55,
        f"throttle {float(action[0]):.3f}",
    )

    # Fresh controller so the integral starts clean for the balloon case.
    ctrl = create_controller("SN65", "KAAO", cruise_altitude_ft=5500.0)
    ctrl.phase = XCPhase.TURN_TO_INTERCEPT
    balloon = _make_state(5500.0 + 370.0, 135.0, heading_deg=40.0,
                          climb_fpm=500.0, roll_deg=30.0)
    action = ctrl.compute_action(balloon, 0.0)
    _check(
        "high/fast turn cuts throttle",
        float(action[0]) <= 0.35,
        f"throttle {float(action[0]):.3f} (old law was fixed 0.70)",
    )
    # Internal elevator: +ve = nose down. theta is 0, target is nose-down.
    _check(
        "high/fast turn commands nose down",
        float(action[2]) > 0.1,
        f"elevator {float(action[2]):.3f} (internal +ve = nose down)",
    )
    _check(
        "turn target altitude stays cruise",
        abs(ctrl.get_target_altitude_ft() - 5500.0) < 1.0,
        f"target {ctrl.get_target_altitude_ft():.0f} ft",
    )
    _check(
        "turn target heading is the runway",
        abs(((ctrl.get_target_heading_deg() - 180.0 + 180) % 360) - 180) < 1.0,
        f"target {ctrl.get_target_heading_deg():.1f}°",
    )


def test_aircraft_database_wiring():
    print("\nAircraft database wiring")
    params, info = resolve_aircraft_params("cessna172")
    db = get_aircraft("cessna172")
    inline = AircraftParams()
    _check(
        "cessna172 comes from the database",
        info["source"] == "aircraft_database"
        and abs(params.latdi.Clb - db.latdi.Clb) < 1e-9
        and abs(params.latdi.Clda - db.latdi.Clda) < 1e-9,
        f"Clb {params.latdi.Clb:.4f} Clda {params.latdi.Clda:.4f} source {info['source']}",
    )
    udaan = get_aircraft("udaan")
    _check(
        "cessna lateral set is the trainer model, not the Udaan table",
        abs(params.latdi.Cnp - inline.latdi.Cnp) < 1e-6
        and abs(params.latdi.Clda - inline.latdi.Clda) < 1e-6
        and abs(params.latdi.Cnp - udaan.latdi.Cnp) > 0.2
        and abs(params.latdi.Clda - udaan.latdi.Clda) > 0.2,
        f"cessna Cnp {params.latdi.Cnp:.4f} Clda {params.latdi.Clda:.4f}; "
        f"udaan Cnp {udaan.latdi.Cnp:.4f} Clda {udaan.latdi.Clda:.4f}",
    )
    # Round-trip matches the converter the simulator uses.
    converted = aircraft_params_from_config(db)
    _check(
        "converter matches resolve_aircraft_params",
        abs(converted.latdi.Cnb - params.latdi.Cnb) < 1e-9
        and abs(converted.mass.mass - params.mass.mass) < 1e-6,
        f"mass {params.mass.mass:.1f} kg",
    )

    fallback, fb_info = resolve_aircraft_params("not-a-packscale-airframe")
    _check(
        "unknown model falls back",
        fb_info["source"] == "AircraftParams_defaults"
        and abs(fallback.latdi.Clb - inline.latdi.Clb) < 1e-9,
        f"source {fb_info['source']} Clb {fallback.latdi.Clb:.4f}",
    )
    empty, empty_info = resolve_aircraft_params("  ")
    _check(
        "blank model id uses cessna172",
        empty_info["model"] == "cessna172" and empty_info["source"] == "aircraft_database",
        f"model {empty_info['model']} via {empty_info['source']}",
    )


def test_start_flight_protocol():
    print("\nstart_flight protocol")
    cmd = {
        "type": "start_flight",
        "origin": "SN65",
        "destination": "KAAO",
        "aircraft": "cessna172",
        "controller_params": {"tp_distance_nm": 10.0},
    }
    model = aircraft_model_from_command(cmd, "cessna172")
    _check(
        "start_flight aircraft field sets the model",
        model == "cessna172" and cmd["origin"] == "SN65" and cmd["destination"] == "KAAO",
        f"model {model} {cmd['origin']}→{cmd['destination']}",
    )
    kept = aircraft_model_from_command(
        {"type": "start_flight", "origin": "SN65", "destination": "K50K"},
        "cessna172",
    )
    _check(
        "start_flight without aircraft keeps the model",
        kept == "cessna172",
        f"model {kept}",
    )
    override_model = aircraft_model_from_command(
        {"type": "override", "action": "set_heading", "value": 90},
        "cessna172",
    )
    _check(
        "override payload does not clear the model",
        override_model == "cessna172",
        f"model {override_model}",
    )
    switched = aircraft_model_from_command(
        {"type": "start_flight", "origin": "SN65", "destination": "KAAO", "model": "udaan"},
        "cessna172",
    )
    _check(
        "model alias is accepted",
        switched == "udaan",
        f"model {switched}",
    )


def test_closed_loop_turn():
    print("\nClosed-loop TURN_TO_INTERCEPT (database Cessna)")
    params, info = resolve_aircraft_params("cessna172")
    assert info["source"] == "aircraft_database"
    dt = 0.02
    sim = FlightSimulator(n_instances=1, params=params, dt=dt, use_gpu=False)
    ctrl = create_controller("SN65", "KAAO", cruise_altitude_ft=5500.0)

    cruise_mps = 110.0 * KTS_TO_MPS
    state0 = np.zeros((1, 12), dtype=np.float32)
    state0[0, StateIndex.Z] = -(5500.0 * FT_TO_M)
    state0[0, StateIndex.U] = cruise_mps
    state0[0, StateIndex.PSI] = ctrl.cruise_heading
    sim.reset(state0)

    ctrl.phase = XCPhase.CRUISE_TO_TP
    ctrl.has_turned_to_cruise = True

    def step_once(sim_time):
        state = sim.get_states()[0]
        action = ctrl.compute_action(state, sim_time)
        sim.set_controls(action.reshape(1, -1))
        sim.step()
        st = sim.get_states()[0]
        alt_ft = -st[StateIndex.Z] * M_TO_FT
        spd_kts = float(np.linalg.norm(st[3:6]) * 1.94384)
        lf = float(sim.get_load_factor()[0])
        return alt_ft, spd_kts, lf

    # Trim in cruise before the turn so the measurement is the turn, not
    # the initial off-trim transient.
    for i in range(int(25.0 / dt)):
        step_once(i * dt)
    alt0, spd0, lf0 = step_once(25.0)
    print(f"  cruise trim: alt {alt0:.0f} ft, spd {spd0:.1f} kt, n {lf0:.2f}")

    ctrl.phase = XCPhase.TURN_TO_INTERCEPT
    alts, spds, lfs = [], [], []
    sim_time = 25.0
    for _ in range(int(50.0 / dt)):
        sim_time += dt
        alt_ft, spd_kts, lf = step_once(sim_time)
        if ctrl.phase != XCPhase.TURN_TO_INTERCEPT:
            break
        alts.append(alt_ft)
        spds.append(spd_kts)
        lfs.append(lf)

    _check(
        "turn lasted long enough to bank",
        len(alts) > int(5.0 / dt),
        f"{len(alts) * dt:.1f} s in TURN_TO_INTERCEPT, ended in {ctrl.phase.name}",
    )
    alt_lo, alt_hi = min(alts), max(alts)
    spd_lo, spd_hi = min(spds), max(spds)
    # Live flight ballooned ~+370 ft and ran ~120→135 kt on the old law.
    _check(
        "turn altitude stays near the cruise trim",
        abs(alt_hi - alt0) < 200.0 and abs(alt_lo - alt0) < 200.0,
        f"trim {alt0:.0f} ft, turn {alt_lo:.0f}..{alt_hi:.0f} ft",
    )
    _check(
        "turn airspeed does not spike",
        spd_hi < 125.0 and spd_lo > 90.0,
        f"trim {spd0:.1f} kt, turn {spd_lo:.1f}..{spd_hi:.1f} kt",
    )
    # Skip the roll-in transient; a steady 30° bank is about 1.15 g.
    steady = lfs[int(4.0 / dt):int(15.0 / dt)] or lfs
    median_n = float(np.median(steady))
    _check(
        "load factor shows the bank, not a stuck 1.0",
        1.08 <= median_n <= 1.35 and max(lfs) < 2.5,
        f"steady median n {median_n:.2f}, span {min(lfs):.2f}..{max(lfs):.2f}",
    )


def test_takeoff_reaches_climb():
    """SN65 ground roll through INITIAL_CLIMB on the database Cessna.

    The live re-fly died here: peak ~167 ft / ~85 kt, heading ran away,
    energy collapsed, and the phase never left INITIAL_CLIMB. That was
    the Udaan lateral block (especially Cnp) on the Cessna, with no CBF
    in the loop. This check uses the same initial condition as
    run_dynamic_xc (viewer runway heading, 400 m behind the threshold).
    """
    print("\nDatabase Cessna takeoff SN65 → climb")
    params, info = resolve_aircraft_params("cessna172")
    assert info["source"] == "aircraft_database"
    origin = KANSAS_AIRPORTS["SN65"]
    ctrl = create_controller("SN65", "KAAO", cruise_altitude_ft=5500.0)
    dt = 0.02
    sim = FlightSimulator(n_instances=1, params=params, dt=dt, use_gpu=False)

    viewer_hdg = np.deg2rad(origin.get_viewer_heading_deg())
    back = viewer_hdg + np.pi
    start_offset_m = 400.0
    state0 = np.zeros((1, 12), dtype=np.float32)
    state0[0, StateIndex.X] = start_offset_m * np.cos(back)
    state0[0, StateIndex.Y] = start_offset_m * np.sin(back)
    state0[0, StateIndex.U] = 5.0
    state0[0, StateIndex.PSI] = viewer_hdg
    sim.reset(state0)

    peak_alt = 0.0
    peak_spd = 0.0
    max_bank = 0.0
    saw_initial = False
    t = 0.0
    end_alt = end_spd = end_hdg = 0.0
    for _ in range(int(90.0 / dt)):
        state = sim.get_states()[0]
        action = ctrl.compute_action(state.copy(), t)
        sim.set_controls(action.reshape(1, -1))
        sim.step()
        st = sim.get_states()[0]
        alt = float(-st[StateIndex.Z] * M_TO_FT)
        spd = float(np.linalg.norm(st[3:6]) * 1.94384)
        bank = abs(float(np.rad2deg(st[StateIndex.PHI])))
        end_hdg = float(np.rad2deg(st[StateIndex.PSI]) % 360)
        peak_alt = max(peak_alt, alt)
        peak_spd = max(peak_spd, spd)
        max_bank = max(max_bank, bank)
        end_alt, end_spd = alt, spd
        if ctrl.phase == XCPhase.INITIAL_CLIMB:
            saw_initial = True
        if ctrl.phase == XCPhase.CLIMB and alt > 1500.0:
            break
        t += dt

    # Heading error from the cruise course, wrapped to [-180, 180].
    hdg_err = (end_hdg - np.rad2deg(ctrl.cruise_heading) + 180.0) % 360.0 - 180.0
    _check(
        "takeoff enters INITIAL_CLIMB and then CLIMB",
        saw_initial and ctrl.phase == XCPhase.CLIMB and end_alt > 1500.0,
        f"phase {ctrl.phase.name} alt {end_alt:.0f} ft at t={t:.1f}s "
        f"(peak {peak_alt:.0f} ft / {peak_spd:.0f} kt)",
    )
    _check(
        "climb holds energy and heading",
        end_spd > 90.0 and max_bank < 45.0 and abs(hdg_err) < 25.0,
        f"spd {end_spd:.1f} kt, max bank {max_bank:.1f}°, "
        f"hdg {end_hdg:.1f}° (cruise {np.rad2deg(ctrl.cruise_heading):.1f}°, err {hdg_err:.1f}°)",
    )


def main():
    test_turn_point_scaling()
    test_turn_energy_hold_command()
    test_aircraft_database_wiring()
    test_start_flight_protocol()
    test_takeoff_reaches_climb()
    test_closed_loop_turn()
    print("\nAll XC quality smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

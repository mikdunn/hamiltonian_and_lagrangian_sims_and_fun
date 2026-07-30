"""Interactive night-sky map for Ann Arbor with Hamiltonian/Lagrangian propagation.

Default coordinates are set to the University of Michigan campus in Ann Arbor.
Users can override location and date using command-line arguments.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


AU_KM = 149_597_870.7
MU_SUN_KM3_S2 = 1.32712440018e11
DAY_SECONDS = 86_400.0
J2000_UTC = datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)
OBLIQUITY_RAD = math.radians(23.43929111)

DEFAULT_LATITUDE_DEG = 42.2780
DEFAULT_LONGITUDE_DEG = -83.7382
DEFAULT_TIMEZONE = "America/Detroit"


@dataclass(frozen=True)
class Star:
    name: str
    ra_hours: float
    dec_deg: float
    magnitude: float


@dataclass(frozen=True)
class PlanetElements:
    name: str
    a_au: float
    e: float
    i_deg: float
    long_node_deg: float
    long_peri_deg: float
    mean_long_deg: float


BRIGHT_STARS = [
    Star("Polaris", 2.5303, 89.2641, 1.97),
    Star("Sirius", 6.7525, -16.7161, -1.46),
    Star("Canopus", 6.3992, -52.6957, -0.74),
    Star("Arcturus", 14.2610, 19.1824, -0.05),
    Star("Vega", 18.6156, 38.7837, 0.03),
    Star("Capella", 5.2782, 45.9980, 0.08),
    Star("Rigel", 5.2423, -8.2016, 0.13),
    Star("Procyon", 7.6550, 5.2250, 0.34),
    Star("Betelgeuse", 5.9195, 7.4071, 0.42),
    Star("Altair", 19.8464, 8.8683, 0.76),
    Star("Aldebaran", 4.5987, 16.5093, 0.85),
    Star("Spica", 13.4199, -11.1614, 0.98),
    Star("Antares", 16.4901, -26.4320, 1.06),
    Star("Pollux", 7.7553, 28.0262, 1.14),
    Star("Fomalhaut", 22.9608, -29.6222, 1.16),
    Star("Deneb", 20.6905, 45.2803, 1.25),
    Star("Regulus", 10.1395, 11.9672, 1.35),
    Star("Adhara", 6.9771, -28.9721, 1.50),
    Star("Castor", 7.5767, 31.8883, 1.58),
    Star("Bellatrix", 5.4189, 6.3497, 1.64),
    Star("Alnitak", 5.6793, -1.9426, 1.74),
    Star("Alnilam", 5.6036, -1.2019, 1.69),
    Star("Mintaka", 5.5334, -0.2991, 2.25),
    Star("Dubhe", 11.0621, 61.7510, 1.79),
    Star("Merak", 11.0307, 56.3823, 2.37),
    Star("Phecda", 11.8972, 53.6948, 2.44),
    Star("Megrez", 12.2570, 57.0326, 3.32),
    Star("Alioth", 12.9004, 55.9598, 1.76),
    Star("Mizar", 13.3988, 54.9254, 2.23),
    Star("Alkaid", 13.7923, 49.3133, 1.85),
    Star("Caph", 0.1529, 59.1498, 2.28),
    Star("Schedar", 0.6751, 56.5373, 2.24),
    Star("GammaCas", 0.9451, 60.7167, 2.15),
    Star("Ruchbah", 1.4303, 60.2353, 2.68),
    Star("Segin", 2.2939, 63.6700, 3.35),
    Star("Denebola", 11.8177, 14.5721, 2.14),
    Star("Sadr", 20.3705, 40.2567, 2.23),
    Star("Albireo", 19.5120, 27.9597, 3.05),
    Star("Elnath", 5.4382, 28.6075, 1.65),
    Star("Saiph", 5.7959, -9.6696, 2.06),
]

CONSTELLATION_EDGES = {
    "Orion": [
        ("Betelgeuse", "Bellatrix"),
        ("Bellatrix", "Mintaka"),
        ("Mintaka", "Alnilam"),
        ("Alnilam", "Alnitak"),
        ("Alnitak", "Saiph"),
        ("Saiph", "Rigel"),
        ("Rigel", "Mintaka"),
        ("Betelgeuse", "Alnitak"),
    ],
    "Ursa Major": [
        ("Dubhe", "Merak"),
        ("Merak", "Phecda"),
        ("Phecda", "Megrez"),
        ("Megrez", "Alioth"),
        ("Alioth", "Mizar"),
        ("Mizar", "Alkaid"),
    ],
    "Cassiopeia": [
        ("Caph", "Schedar"),
        ("Schedar", "GammaCas"),
        ("GammaCas", "Ruchbah"),
        ("Ruchbah", "Segin"),
    ],
    "Cygnus": [
        ("Deneb", "Sadr"),
        ("Sadr", "Albireo"),
    ],
    "Gemini": [
        ("Castor", "Pollux"),
    ],
    "Taurus": [
        ("Aldebaran", "Elnath"),
    ],
    "Leo": [
        ("Regulus", "Denebola"),
    ],
}

PLANET_ELEMENTS = [
    PlanetElements("Mercury", 0.38709927, 0.20563593, 7.00497902, 48.33076593, 77.45779628, 252.25032350),
    PlanetElements("Venus", 0.72333566, 0.00677672, 3.39467605, 76.67984255, 131.60246718, 181.97909950),
    PlanetElements("Earth", 1.00000261, 0.01671123, 0.00001531, -11.26064, 102.93768193, 100.46457166),
    PlanetElements("Mars", 1.52371034, 0.09339410, 1.84969142, 49.55953891, 336.04084, -4.55343205),
    PlanetElements("Jupiter", 5.20288700, 0.04838624, 1.30439695, 100.47390909, 14.72847983, 34.39644051),
    PlanetElements("Saturn", 9.53667594, 0.05386179, 2.48599187, 113.66242448, 92.59887831, 49.95424423),
    PlanetElements("Uranus", 19.18916464, 0.04725744, 0.77263783, 74.01692503, 170.95427630, 313.23810451),
    PlanetElements("Neptune", 30.06992276, 0.00859048, 1.77004347, 131.78422574, 44.96476227, -55.12002969),
]

FEATURED_STAR_DESCRIPTIONS = {
    "Polaris": "North Star near true north",
    "Sirius": "brightest night-sky star",
    "Vega": "Summer Triangle anchor",
    "Arcturus": "orange giant in Bootes",
    "Capella": "bright yellow giant in Auriga",
    "Betelgeuse": "red supergiant in Orion",
    "Rigel": "blue supergiant in Orion",
    "Aldebaran": "eye of Taurus",
    "Antares": "heart of Scorpius",
    "Deneb": "Summer Triangle anchor in Cygnus",
    "Altair": "Summer Triangle anchor in Aquila",
}

CONSTELLATION_DESCRIPTIONS = {
    "Orion": "The Hunter",
    "Ursa Major": "The Great Bear",
    "Cassiopeia": "The Queen",
    "Cygnus": "The Swan",
    "Gemini": "The Twins",
    "Taurus": "The Bull",
    "Leo": "The Lion",
}

PLANET_DESCRIPTIONS = {
    "Mercury": "inner planet near twilight",
    "Venus": "very bright evening/morning planet",
    "Mars": "reddish wandering planet",
    "Jupiter": "largest planet, bright with steady glow",
    "Saturn": "golden giant with rings",
    "Uranus": "ice giant (often binocular target)",
    "Neptune": "faint ice giant",
    "Sun": "solar reference body",
}


def wrap_angle_rad(theta: float) -> float:
    return theta % (2.0 * math.pi)


def julian_day(dt_utc: datetime) -> float:
    dt_utc = dt_utc.astimezone(UTC)
    year = dt_utc.year
    month = dt_utc.month
    day = dt_utc.day
    hour = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0 + dt_utc.microsecond / 3.6e9

    if month <= 2:
        year -= 1
        month += 12

    a = year // 100
    b = 2 - a + (a // 4)
    jd0 = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
    return jd0 + hour / 24.0


def local_sidereal_time_rad(dt_utc: datetime, longitude_deg: float) -> float:
    jd = julian_day(dt_utc)
    t = (jd - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * (t * t)
        - (t * t * t) / 38710000.0
    )
    lst_deg = (gmst_deg + longitude_deg) % 360.0
    return math.radians(lst_deg)


def ra_dec_to_alt_az(ra_rad: np.ndarray, dec_rad: np.ndarray, latitude_deg: float, lst_rad: float) -> tuple[np.ndarray, np.ndarray]:
    lat = math.radians(latitude_deg)
    hour_angle = (lst_rad - ra_rad + np.pi) % (2.0 * np.pi) - np.pi

    sin_alt = np.sin(dec_rad) * math.sin(lat) + np.cos(dec_rad) * math.cos(lat) * np.cos(hour_angle)
    sin_alt = np.clip(sin_alt, -1.0, 1.0)
    alt = np.arcsin(sin_alt)

    y = -np.sin(hour_angle)
    x = np.tan(dec_rad) * math.cos(lat) - math.sin(lat) * np.cos(hour_angle)
    az = np.arctan2(y, x)
    az = np.mod(az, 2.0 * np.pi)
    return alt, az


def sky_projection(alt_rad: np.ndarray, az_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = (0.5 * np.pi - alt_rad) / (0.5 * np.pi)
    x = radius * np.sin(az_rad)
    y = radius * np.cos(az_rad)
    return x, y


def solve_kepler_equation(mean_anomaly_rad: float, eccentricity: float) -> float:
    anomaly = mean_anomaly_rad if eccentricity < 0.8 else math.pi
    for _ in range(20):
        f = anomaly - eccentricity * math.sin(anomaly) - mean_anomaly_rad
        fp = 1.0 - eccentricity * math.cos(anomaly)
        anomaly_next = anomaly - f / fp
        if abs(anomaly_next - anomaly) < 1e-12:
            anomaly = anomaly_next
            break
        anomaly = anomaly_next
    return anomaly


def elements_to_state_vectors(elements: PlanetElements, epoch_utc: datetime) -> tuple[np.ndarray, np.ndarray]:
    a_km = elements.a_au * AU_KM
    i_rad = math.radians(elements.i_deg)
    omega_node = math.radians(elements.long_node_deg)
    arg_peri = math.radians(elements.long_peri_deg - elements.long_node_deg)
    mean_anomaly_0 = math.radians(elements.mean_long_deg - elements.long_peri_deg)

    delta_t = (epoch_utc - J2000_UTC).total_seconds()
    mean_motion = math.sqrt(MU_SUN_KM3_S2 / (a_km**3))
    mean_anomaly = wrap_angle_rad(mean_anomaly_0 + mean_motion * delta_t)
    ecc_anomaly = solve_kepler_equation(mean_anomaly, elements.e)

    cos_e = math.cos(ecc_anomaly)
    sin_e = math.sin(ecc_anomaly)
    sqrt_one_minus_e2 = math.sqrt(1.0 - elements.e * elements.e)

    x_pf = a_km * (cos_e - elements.e)
    y_pf = a_km * (sqrt_one_minus_e2 * sin_e)
    radius = a_km * (1.0 - elements.e * cos_e)

    factor = math.sqrt(MU_SUN_KM3_S2 * a_km) / radius
    vx_pf = -factor * sin_e
    vy_pf = factor * sqrt_one_minus_e2 * cos_e

    c_o = math.cos(omega_node)
    s_o = math.sin(omega_node)
    c_w = math.cos(arg_peri)
    s_w = math.sin(arg_peri)
    c_i = math.cos(i_rad)
    s_i = math.sin(i_rad)

    r = np.array(
        [
            (c_o * c_w - s_o * s_w * c_i) * x_pf + (-c_o * s_w - s_o * c_w * c_i) * y_pf,
            (s_o * c_w + c_o * s_w * c_i) * x_pf + (-s_o * s_w + c_o * c_w * c_i) * y_pf,
            (s_w * s_i) * x_pf + (c_w * s_i) * y_pf,
        ],
        dtype=float,
    )
    v = np.array(
        [
            (c_o * c_w - s_o * s_w * c_i) * vx_pf + (-c_o * s_w - s_o * c_w * c_i) * vy_pf,
            (s_o * c_w + c_o * s_w * c_i) * vx_pf + (-s_o * s_w + c_o * c_w * c_i) * vy_pf,
            (s_w * s_i) * vx_pf + (c_w * s_i) * vy_pf,
        ],
        dtype=float,
    )
    return r, v


def gravitational_acceleration(position_km: np.ndarray) -> np.ndarray:
    radius = float(np.linalg.norm(position_km))
    return -MU_SUN_KM3_S2 * position_km / (radius**3)


def leapfrog_step(position_km: np.ndarray, velocity_km_s: np.ndarray, dt_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    a0 = gravitational_acceleration(position_km)
    v_half = velocity_km_s + 0.5 * dt_seconds * a0
    r_next = position_km + dt_seconds * v_half
    a1 = gravitational_acceleration(r_next)
    v_next = v_half + 0.5 * dt_seconds * a1
    return r_next, v_next


def ecliptic_to_equatorial(vector_xyz: np.ndarray) -> np.ndarray:
    x, y, z = vector_xyz
    cos_eps = math.cos(OBLIQUITY_RAD)
    sin_eps = math.sin(OBLIQUITY_RAD)
    return np.array([x, y * cos_eps - z * sin_eps, y * sin_eps + z * cos_eps], dtype=float)


def cartesian_to_ra_dec(vector_xyz: np.ndarray) -> tuple[float, float]:
    x, y, z = vector_xyz
    ra = wrap_angle_rad(math.atan2(y, x))
    dec = math.atan2(z, math.sqrt(x * x + y * y))
    return ra, dec


def specific_hamiltonian(position_km: np.ndarray, velocity_km_s: np.ndarray) -> float:
    radius = float(np.linalg.norm(position_km))
    kinetic = 0.5 * float(np.dot(velocity_km_s, velocity_km_s))
    potential = -MU_SUN_KM3_S2 / radius
    return kinetic + potential


def specific_lagrangian(position_km: np.ndarray, velocity_km_s: np.ndarray) -> float:
    radius = float(np.linalg.norm(position_km))
    kinetic = 0.5 * float(np.dot(velocity_km_s, velocity_km_s))
    potential = -MU_SUN_KM3_S2 / radius
    return kinetic - potential


def build_planet_tracks(start_utc: datetime, n_steps: int, step_minutes: int) -> tuple[list[datetime], dict[str, np.ndarray], dict[str, float]]:
    times_utc = [start_utc + timedelta(minutes=step_minutes * i) for i in range(n_steps)]
    dt_seconds = step_minutes * 60.0

    elements_by_name = {elem.name: elem for elem in PLANET_ELEMENTS}
    states: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    trajectories: dict[str, np.ndarray] = {}

    for elem in PLANET_ELEMENTS:
        r0, v0 = elements_to_state_vectors(elem, start_utc)
        states[elem.name] = (r0, v0)
        trajectories[elem.name] = np.zeros((n_steps, 3), dtype=float)

    earth_h0 = specific_hamiltonian(states["Earth"][0], states["Earth"][1])

    for step_idx in range(n_steps):
        for name in elements_by_name:
            trajectories[name][step_idx] = states[name][0]
        if step_idx == n_steps - 1:
            break
        for name in elements_by_name:
            r_now, v_now = states[name]
            states[name] = leapfrog_step(r_now, v_now, dt_seconds)

    earth_final_state = states["Earth"]
    earth_h1 = specific_hamiltonian(earth_final_state[0], earth_final_state[1])
    earth_l0 = specific_lagrangian(trajectories["Earth"][0], elements_to_state_vectors(elements_by_name["Earth"], start_utc)[1])
    earth_l1 = specific_lagrangian(earth_final_state[0], earth_final_state[1])

    diagnostics = {
        "earth_hamiltonian_initial": earth_h0,
        "earth_hamiltonian_final": earth_h1,
        "earth_hamiltonian_drift": earth_h1 - earth_h0,
        "earth_lagrangian_initial": earth_l0,
        "earth_lagrangian_final": earth_l1,
        "earth_lagrangian_drift": earth_l1 - earth_l0,
    }

    return times_utc, trajectories, diagnostics


def compute_planet_radec(trajectories: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    earth_track = trajectories["Earth"]
    radec: dict[str, np.ndarray] = {}

    for name, track in trajectories.items():
        if name == "Earth":
            continue
        geocentric = track - earth_track
        angles = np.zeros((track.shape[0], 2), dtype=float)
        for idx in range(track.shape[0]):
            equatorial = ecliptic_to_equatorial(geocentric[idx])
            angles[idx] = cartesian_to_ra_dec(equatorial)
        radec[name] = angles

    sun_geocentric = -earth_track
    sun_angles = np.zeros((earth_track.shape[0], 2), dtype=float)
    for idx in range(earth_track.shape[0]):
        sun_angles[idx] = cartesian_to_ra_dec(ecliptic_to_equatorial(sun_geocentric[idx]))
    radec["Sun"] = sun_angles
    return radec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Night-sky simulator with planetary dynamics.")
    parser.add_argument("--lat", type=float, default=DEFAULT_LATITUDE_DEG, help="Observer latitude in degrees.")
    parser.add_argument("--lon", type=float, default=DEFAULT_LONGITUDE_DEG, help="Observer longitude in degrees.")
    parser.add_argument("--date", type=str, default=None, help="Local date as YYYY-MM-DD. Default uses current local date.")
    parser.add_argument("--start-hour", type=float, default=20.0, help="Local start hour for the simulation window.")
    parser.add_argument("--duration-hours", type=float, default=10.0, help="Duration of the night simulation window.")
    parser.add_argument("--step-minutes", type=int, default=10, help="Time resolution of simulation in minutes.")
    parser.add_argument("--timezone", type=str, default=DEFAULT_TIMEZONE, help="IANA timezone string.")
    parser.add_argument("--no-ui", action="store_true", help="Print visible objects instead of opening an interactive plot.")
    parser.add_argument("--save-frame", type=str, default=None, help="Optional image path for saving the initial map frame.")
    return parser.parse_args()


def build_start_time_local(args: argparse.Namespace) -> datetime:
    tz = ZoneInfo(args.timezone)
    if args.date is None:
        target_date = datetime.now(tz).date()
    else:
        target_date = date.fromisoformat(args.date)

    hour = int(args.start_hour)
    minute = int(round((args.start_hour - hour) * 60.0))
    return datetime(target_date.year, target_date.month, target_date.day, hour, minute, tzinfo=tz)


def run_interactive_simulator(args: argparse.Namespace) -> None:
    start_local = build_start_time_local(args)
    start_utc = start_local.astimezone(UTC)
    n_steps = max(2, int(round(args.duration_hours * 60.0 / args.step_minutes)) + 1)

    times_utc, trajectories, diagnostics = build_planet_tracks(start_utc=start_utc, n_steps=n_steps, step_minutes=args.step_minutes)
    planet_radec = compute_planet_radec(trajectories)

    star_names = [star.name for star in BRIGHT_STARS]
    star_ra = np.array([math.radians(star.ra_hours * 15.0) for star in BRIGHT_STARS], dtype=float)
    star_dec = np.array([math.radians(star.dec_deg) for star in BRIGHT_STARS], dtype=float)
    star_sizes = np.array([max(8.0, 80.0 - 16.0 * star.magnitude) for star in BRIGHT_STARS], dtype=float)
    name_to_star_idx = {name: idx for idx, name in enumerate(star_names)}

    constellation_sets = {
        name: {s for edge in edges for s in edge}
        for name, edges in CONSTELLATION_EDGES.items()
    }

    fig = plt.figure(figsize=(13, 8), facecolor="#070b1a")
    ax_sky = fig.add_axes([0.06, 0.16, 0.62, 0.78])
    ax_info = fig.add_axes([0.72, 0.16, 0.25, 0.78])
    ax_slider = fig.add_axes([0.08, 0.06, 0.84, 0.045], facecolor="#1b223d")

    ax_sky.set_facecolor("#070b1a")
    ax_sky.set_aspect("equal")
    ax_sky.set_xlim(-1.08, 1.08)
    ax_sky.set_ylim(-1.08, 1.08)
    ax_sky.axis("off")

    theta = np.linspace(0.0, 2.0 * np.pi, 512)
    ax_sky.plot(np.sin(theta), np.cos(theta), color="#6ec1ff", linewidth=1.5, alpha=0.8)
    ax_sky.text(0.0, 1.03, "N", color="#c4d9ff", ha="center", va="bottom", fontsize=11)
    ax_sky.text(1.03, 0.0, "E", color="#c4d9ff", ha="left", va="center", fontsize=11)
    ax_sky.text(0.0, -1.03, "S", color="#c4d9ff", ha="center", va="top", fontsize=11)
    ax_sky.text(-1.03, 0.0, "W", color="#c4d9ff", ha="right", va="center", fontsize=11)
    ax_sky.text(0.0, 0.0, "Zenith", color="#87b4ff", ha="center", va="center", fontsize=10, alpha=0.85)

    star_scatter = ax_sky.scatter([], [], s=[], c="#f6f7ff", alpha=0.9, edgecolors="none")
    polaris_scatter = ax_sky.scatter([], [], s=[], c="#8ed8ff", alpha=0.95, edgecolors="#d7f2ff", linewidths=0.9)
    planet_scatter = ax_sky.scatter([], [], s=[], c="#ffbb45", alpha=0.95, edgecolors="#ffd88c", linewidths=0.7)

    constellation_lines = []
    for _name, edges in CONSTELLATION_EDGES.items():
        for _ in edges:
            line, = ax_sky.plot([], [], color="#3a6fb0", linewidth=1.0, alpha=0.65)
            constellation_lines.append(line)

    ax_info.set_facecolor("#10162a")
    ax_info.axis("off")

    info_text = ax_info.text(
        0.02,
        0.98,
        "",
        transform=ax_info.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="#e3ecff",
        family="monospace",
        linespacing=1.35,
    )

    slider = Slider(
        ax=ax_slider,
        label="Time step",
        valmin=0,
        valmax=n_steps - 1,
        valinit=0,
        valstep=1,
        color="#60b8ff",
    )
    slider.label.set_color("#d7e8ff")

    dynamic_labels: list[plt.Text] = []
    edge_pairs = [(c_name, edge) for c_name, edges in CONSTELLATION_EDGES.items() for edge in edges]
    polaris_idx = name_to_star_idx.get("Polaris")

    def update(step_float: float) -> None:
        nonlocal dynamic_labels
        idx = int(step_float)
        when_utc = times_utc[idx]
        when_local = when_utc.astimezone(ZoneInfo(args.timezone))
        lst = local_sidereal_time_rad(when_utc, args.lon)

        alt_star, az_star = ra_dec_to_alt_az(star_ra, star_dec, args.lat, lst)
        visible_stars = alt_star > 0.0
        x_star, y_star = sky_projection(alt_star[visible_stars], az_star[visible_stars])

        star_offsets = np.column_stack([x_star, y_star]) if np.any(visible_stars) else np.zeros((0, 2))
        star_scatter.set_offsets(star_offsets)
        star_scatter.set_sizes(star_sizes[visible_stars])

        if polaris_idx is not None and visible_stars[polaris_idx]:
            pol_alt = np.array([alt_star[polaris_idx]], dtype=float)
            pol_az = np.array([az_star[polaris_idx]], dtype=float)
            pol_x, pol_y = sky_projection(pol_alt, pol_az)
            polaris_scatter.set_offsets(np.array([[pol_x[0], pol_y[0]]], dtype=float))
            polaris_scatter.set_sizes(np.array([160.0], dtype=float))
        else:
            polaris_scatter.set_offsets(np.zeros((0, 2), dtype=float))
            polaris_scatter.set_sizes(np.zeros((0,), dtype=float))

        for text_item in dynamic_labels:
            text_item.remove()
        dynamic_labels = []

        for star_idx in np.where(visible_stars)[0]:
            star_name = star_names[star_idx]
            should_label = star_name in FEATURED_STAR_DESCRIPTIONS or BRIGHT_STARS[star_idx].magnitude <= 1.0
            if not should_label:
                continue
            sx, sy = sky_projection(np.array([alt_star[star_idx]]), np.array([az_star[star_idx]]))
            display_name = star_name
            if star_name == "Polaris":
                display_name = "Polaris (North Star)"
            label = ax_sky.text(sx[0] + 0.015, sy[0] + 0.015, display_name, color="#dfe9ff", fontsize=8)
            dynamic_labels.append(label)

        planet_positions = []
        planet_sizes = []
        visible_planets = []
        planet_label_color = {
            "Mercury": "#ffc078",
            "Venus": "#ffe08f",
            "Mars": "#ff9d7f",
            "Jupiter": "#ffd4a3",
            "Saturn": "#ffe8b3",
            "Uranus": "#a9e8ff",
            "Neptune": "#8cb8ff",
            "Sun": "#ffe36b",
        }

        for name, angles in planet_radec.items():
            ra = np.array([angles[idx, 0]], dtype=float)
            dec = np.array([angles[idx, 1]], dtype=float)
            alt, az = ra_dec_to_alt_az(ra, dec, args.lat, lst)
            if alt[0] <= 0.0:
                continue
            px, py = sky_projection(alt, az)
            planet_positions.append([px[0], py[0]])
            planet_sizes.append(75.0 if name == "Sun" else 48.0)
            visible_planets.append((name, math.degrees(alt[0])))
            text_obj = ax_sky.text(px[0] + 0.02, py[0] - 0.02, name, color=planet_label_color.get(name, "#ffd78a"), fontsize=9)
            dynamic_labels.append(text_obj)

        if planet_positions:
            planet_scatter.set_offsets(np.array(planet_positions, dtype=float))
            planet_scatter.set_sizes(np.array(planet_sizes, dtype=float))
        else:
            planet_scatter.set_offsets(np.zeros((0, 2)))
            planet_scatter.set_sizes(np.zeros((0,), dtype=float))

        for line_artist, (_const_name, (star_a, star_b)) in zip(constellation_lines, edge_pairs):
            idx_a = name_to_star_idx.get(star_a)
            idx_b = name_to_star_idx.get(star_b)
            if idx_a is None or idx_b is None:
                line_artist.set_data([], [])
                continue
            if not (visible_stars[idx_a] and visible_stars[idx_b]):
                line_artist.set_data([], [])
                continue
            xa, ya = sky_projection(np.array([alt_star[idx_a]]), np.array([az_star[idx_a]]))
            xb, yb = sky_projection(np.array([alt_star[idx_b]]), np.array([az_star[idx_b]]))
            line_artist.set_data([xa[0], xb[0]], [ya[0], yb[0]])

        visible_constellations = []
        for constellation_name, star_set in constellation_sets.items():
            visible_count = sum(1 for star_name in star_set if star_name in name_to_star_idx and visible_stars[name_to_star_idx[star_name]])
            if visible_count >= 2:
                visible_constellations.append(constellation_name)

                points = []
                for star_name in star_set:
                    s_idx = name_to_star_idx.get(star_name)
                    if s_idx is None or not visible_stars[s_idx]:
                        continue
                    cx, cy = sky_projection(np.array([alt_star[s_idx]]), np.array([az_star[s_idx]]))
                    points.append((cx[0], cy[0]))
                if points:
                    px = float(np.mean([p[0] for p in points]))
                    py = float(np.mean([p[1] for p in points]))
                    cname = CONSTELLATION_DESCRIPTIONS.get(constellation_name, constellation_name)
                    ctext = ax_sky.text(px, py, f"{constellation_name}\n({cname})", color="#8fc6ff", fontsize=7, ha="center", va="center")
                    dynamic_labels.append(ctext)

        visible_planets.sort(key=lambda item: item[1], reverse=True)
        planet_lines = [
            f"- {name:8s} alt={alt_deg:5.1f} deg | {PLANET_DESCRIPTIONS.get(name, 'planet')}"
            for name, alt_deg in visible_planets
        ]
        if not planet_lines:
            planet_lines = ["- none above horizon"]

        known_star_lines = []
        for star_name, star_desc in FEATURED_STAR_DESCRIPTIONS.items():
            star_idx = name_to_star_idx.get(star_name)
            if star_idx is None or not visible_stars[star_idx]:
                continue
            known_star_lines.append(f"- {star_name}: {star_desc}")
        if not known_star_lines:
            known_star_lines = ["- none"]

        if polaris_idx is not None:
            polaris_alt_deg = math.degrees(alt_star[polaris_idx])
            polaris_status = f"visible at {polaris_alt_deg:.1f} deg altitude" if polaris_alt_deg > 0 else "below horizon"
        else:
            polaris_status = "not in star catalog"

        info_lines = [
            "Observer",
            f"lat={args.lat:.4f} deg",
            f"lon={args.lon:.4f} deg",
            "",
            "Local time",
            when_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "",
            "Visible planets",
            *planet_lines,
            "",
            "Known stars",
            *known_star_lines[:6],
            "",
            f"Polaris status: {polaris_status}",
            "",
            "Visible constellations",
            ", ".join(visible_constellations[:7]) if visible_constellations else "none",
            "",
            "Dynamics check",
            f"dH(Earth)={diagnostics['earth_hamiltonian_drift']:+.2e}",
            f"dL(Earth)={diagnostics['earth_lagrangian_drift']:+.2e}",
        ]
        info_text.set_text("\n".join(info_lines))

        ax_sky.set_title(
            "Ann Arbor Night Sky Map with Hamiltonian/Lagrangian Planet Propagation",
            color="#e6f1ff",
            fontsize=12,
            pad=12,
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)

    if args.save_frame:
        fig.savefig(args.save_frame, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved frame to: {args.save_frame}")

    plt.show()


def run_non_interactive(args: argparse.Namespace) -> None:
    start_local = build_start_time_local(args)
    start_utc = start_local.astimezone(UTC)
    n_steps = max(2, int(round(args.duration_hours * 60.0 / args.step_minutes)) + 1)

    times_utc, trajectories, diagnostics = build_planet_tracks(start_utc=start_utc, n_steps=n_steps, step_minutes=args.step_minutes)
    planet_radec = compute_planet_radec(trajectories)

    star_names = [star.name for star in BRIGHT_STARS]
    star_ra = np.array([math.radians(star.ra_hours * 15.0) for star in BRIGHT_STARS], dtype=float)
    star_dec = np.array([math.radians(star.dec_deg) for star in BRIGHT_STARS], dtype=float)
    name_to_star_idx = {name: idx for idx, name in enumerate(star_names)}
    constellation_sets = {
        name: {s for edge in edges for s in edge}
        for name, edges in CONSTELLATION_EDGES.items()
    }
    print("Night-sky visibility report")
    print(f"Location: lat={args.lat:.4f}, lon={args.lon:.4f}")
    print(f"Window start: {start_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Duration hours: {args.duration_hours:.2f} | step minutes: {args.step_minutes}")
    print(f"Earth Hamiltonian drift: {diagnostics['earth_hamiltonian_drift']:+.3e}")
    print(f"Earth Lagrangian drift: {diagnostics['earth_lagrangian_drift']:+.3e}")
    print()

    for idx in range(0, n_steps, max(1, int(round(60 / args.step_minutes)))):
        dt_utc = times_utc[idx]
        dt_local = dt_utc.astimezone(ZoneInfo(args.timezone))
        lst = local_sidereal_time_rad(dt_utc, args.lon)

        alt_star, _az_star = ra_dec_to_alt_az(star_ra, star_dec, args.lat, lst)
        visible_stars = alt_star > 0.0
        known_star_labels = [
            f"{name}({math.degrees(alt_star[name_to_star_idx[name]]):.1f}deg)"
            for name in FEATURED_STAR_DESCRIPTIONS
            if name in name_to_star_idx and visible_stars[name_to_star_idx[name]]
        ]

        visible_constellations = []
        for constellation_name, star_set in constellation_sets.items():
            visible_count = sum(1 for star_name in star_set if star_name in name_to_star_idx and visible_stars[name_to_star_idx[star_name]])
            if visible_count >= 2:
                visible_constellations.append(constellation_name)

        visible_planets = []
        for name, angles in planet_radec.items():
            ra = np.array([angles[idx, 0]])
            dec = np.array([angles[idx, 1]])
            alt, _az = ra_dec_to_alt_az(ra, dec, args.lat, lst)
            if alt[0] > 0.0:
                visible_planets.append((name, math.degrees(alt[0])))
        visible_planets.sort(key=lambda item: item[1], reverse=True)
        labels = ", ".join(f"{name}({alt_deg:.1f}deg)" for name, alt_deg in visible_planets) if visible_planets else "none"

        known_stars_text = ", ".join(known_star_labels[:6]) if known_star_labels else "none"
        const_text = ", ".join(visible_constellations[:5]) if visible_constellations else "none"
        print(f"{dt_local.strftime('%H:%M')} -> planets: {labels}")
        print(f"          known stars: {known_stars_text}")
        print(f"          constellations: {const_text}")


def main() -> None:
    args = parse_args()
    if args.no_ui:
        run_non_interactive(args)
    else:
        run_interactive_simulator(args)


if __name__ == "__main__":
    main()

# 3D orbital-overlap & bonding explorer (qualitative LCAO /
# extended-Hückel teaching model)
import math
import html as _html
import traceback
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage import measure
from scipy.linalg import eigh

BOHR = 0.529177  # Å
SQ, PI = math.sqrt, math.pi

# shells: (principal n, Slater zeta in bohr^-1); vsip in eV; val = valence
# electrons
ELEMENTS = {
    'H': dict(val=1, rcov=0.31, color='#f4f4f4', size=8,
              shells=dict(s=(1, 1.30), p=(2, 0.80), d=(3, 0.60), f=(4, 0.50)),
              vsip=dict(s=-13.6, p=-3.4, d=-1.5, f=-0.85)),
    'C': dict(val=4, rcov=0.76, color='#555555', size=12,
              shells=dict(s=(2, 1.625), p=(2, 1.625),
                          d=(3, 1.00), f=(4, 0.80)),
              vsip=dict(s=-21.4, p=-11.4, d=-6.0, f=-4.0)),
    'N': dict(val=5, rcov=0.71, color='#3050f8', size=12,
              shells=dict(s=(2, 1.950), p=(2, 1.950),
                          d=(3, 1.10), f=(4, 0.90)),
              vsip=dict(s=-26.0, p=-13.4, d=-6.5, f=-4.5)),
    'O': dict(val=6, rcov=0.66, color='#ff0d0d', size=12,
              shells=dict(s=(2, 2.275), p=(2, 2.275),
                          d=(3, 1.20), f=(4, 0.95)),
              vsip=dict(s=-32.3, p=-14.8, d=-7.0, f=-5.0)),
    'F': dict(val=7, rcov=0.57, color='#90e050', size=12,
              shells=dict(s=(2, 2.425), p=(2, 2.425),
                          d=(3, 1.25), f=(4, 1.00)),
              vsip=dict(s=-40.0, p=-18.1, d=-8.0, f=-5.0)),
    'P': dict(val=5, rcov=1.07, color='#ff8000', size=14,
              shells=dict(s=(3, 1.600), p=(3, 1.600),
                          d=(3, 1.40), f=(4, 1.00)),
              vsip=dict(s=-18.6, p=-14.0, d=-7.0, f=-5.0)),
    'S': dict(val=6, rcov=1.05, color='#e6c200', size=14,
              shells=dict(s=(3, 1.817), p=(3, 1.817),
                          d=(3, 1.50), f=(4, 1.00)),
              vsip=dict(s=-20.0, p=-13.3, d=-8.0, f=-5.0)),
    'Cl': dict(val=7, rcov=1.02, color='#1ff01f', size=14,
               shells=dict(s=(3, 2.033), p=(3, 2.033),
                           d=(3, 1.60), f=(4, 1.10)),
               vsip=dict(s=-26.3, p=-14.2, d=-9.0, f=-5.0)),
}
HB_ATOMS = {'N', 'O', 'F'}

# Real (tesseral) spherical harmonics as functions of the unit vector (X, Y, Z)
YLM = {
    's': lambda X, Y, Z: np.full_like(X, 0.5 / SQ(PI)),
    'px': lambda X, Y, Z: SQ(3 / (4 * PI)) * X,
    'py': lambda X, Y, Z: SQ(3 / (4 * PI)) * Y,
    'pz': lambda X, Y, Z: SQ(3 / (4 * PI)) * Z,
    'dz2': lambda X, Y, Z: 0.25 * SQ(5 / PI) * (3 * Z**2 - 1),
    'dxz': lambda X, Y, Z: 0.5 * SQ(15 / PI) * X * Z,
    'dyz': lambda X, Y, Z: 0.5 * SQ(15 / PI) * Y * Z,
    'dxy': lambda X, Y, Z: 0.5 * SQ(15 / PI) * X * Y,
    'dx2-y2': lambda X, Y, Z: 0.25 * SQ(15 / PI) * (X**2 - Y**2),
    'fz3': lambda X, Y, Z: 0.25 * SQ(7 / PI) * Z * (5 * Z**2 - 3),
    'fxz2': lambda X, Y, Z: 0.25 * SQ(21 / (2 * PI)) * X * (5 * Z**2 - 1),
    'fyz2': lambda X, Y, Z: 0.25 * SQ(21 / (2 * PI)) * Y * (5 * Z**2 - 1),
    'fxyz': lambda X, Y, Z: 0.5 * SQ(105 / PI) * X * Y * Z,
    'fz(x2-y2)': lambda X, Y, Z: 0.25 * SQ(105 / PI) * Z * (X**2 - Y**2),
    'fx(x2-3y2)': lambda X, Y, Z: 0.25 * SQ(35 / (2 * PI)) * X * (X**2 - 3 * Y**2),
    'fy(3x2-y2)': lambda X, Y, Z: 0.25 * SQ(35 / (2 * PI)) * Y * (3 * X**2 - Y**2),
}
PRETTY = {'dz2': 'dz²', 'dx2-y2': 'dx²−y²', 'fz3': 'fz³', 'fxz2': 'fxz²', 'fyz2': 'fyz²',
          'fz(x2-y2)': 'fz(x²−y²)', 'fx(x2-3y2)': 'fx(x²−3y²)', 'fy(3x2-y2)': 'fy(3x²−y²)'}


def _Y(name, u):
    u = np.asarray(u, float)
    return float(YLM[name](*(np.array([c]) for c in u))[0])


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


_T = SQ(2 / 3)
# hybrid set -> (basis functions, lobe directions in the canonical frame,
# unhybridized leftovers)
HYBRIDS = {
    'sp': (['s', 'px'], [(1, 0, 0), (-1, 0, 0)], ['py', 'pz']),
    'sp2': (['s', 'px', 'py'], [(1, 0, 0), (-0.5, SQ(3) / 2, 0), (-0.5, -SQ(3) / 2, 0)], ['pz']),
    'sp3': (['s', 'px', 'py', 'pz'],
            [(1, 0, 0), (-1 / 3, 2 * SQ(2) / 3, 0), (-1 / 3, -SQ(2) / 3, _T), (-1 / 3, -SQ(2) / 3, -_T)], []),
    'sp3d': (['s', 'px', 'py', 'pz', 'dz2'],
             [(1, 0, 0), (-0.5, SQ(3) / 2, 0), (-0.5, -SQ(3) / 2, 0), (0, 0, 1), (0, 0, -1)], []),
    'sp3d2': (['s', 'px', 'py', 'pz', 'dz2', 'dx2-y2'],
              [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], []),
    'sp3d3f': (['s', 'px', 'py', 'pz', 'dxy', 'dxz', 'dyz', 'fxyz'],
               [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)], []),
}
PURE = {'s': ['s'], 'p': ['px', 'py', 'pz'],
        'd': ['dz2', 'dxz', 'dyz', 'dxy', 'dx2-y2'],
        'f': ['fz3', 'fxz2', 'fyz2', 'fxyz', 'fz(x2-y2)', 'fx(x2-3y2)', 'fy(3x2-y2)']}
ORBITAL_SETS = [
    's',
    'p',
    'd',
    'f',
    'sp',
    'sp2',
    'sp3',
    'sp3d',
    'sp3d2',
    'sp3d3f']
SET_LABEL = {'s': 's', 'p': 'p', 'd': 'd', 'f': 'f', 'sp': 'sp', 'sp2': 'sp²', 'sp3': 'sp³',
             'sp3d': 'sp³d', 'sp3d2': 'sp³d²', 'sp3d3f': 'sp³d³f (spdf)'}


def _rot_to_x(u):
    """Rotation matrix taking unit vector u onto +x (Rodrigues)."""
    u = _unit(u)
    x = np.array([1.0, 0, 0])
    v = np.cross(u, x)
    s = np.linalg.norm(v)
    c = float(u @ x)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([-1.0, -1.0, 1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s**2)


# canonical frame -> "aim" frame (primary lobe / bond axis on local +x)
_Z_TO_X = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
QFRAME = {'d': _Z_TO_X, 'f': _Z_TO_X, 'sp3d3f': _rot_to_x((1, 1, 1))}


def set_lobes(oset):
    """Directions (canonical frame) used to place neighbours on an atom's orbital set."""
    if oset in HYBRIDS:
        return [_unit(u) for u in HYBRIDS[oset][1]]
    if oset == 'p':
        return [np.eye(3)[k] for k in range(3)]
    if oset in ('d', 'f'):
        return [np.array([0.0, 0, 1])]
    return [np.array([1.0, 0, 0])]


def build_orbitals(oset):
    if oset in HYBRIDS:
        basis, lobes, rest = HYBRIDS[oset]
        lobes = [_unit(u) for u in lobes]
        A = np.array([[_Y(b, u) for b in basis] for u in lobes])
        w, V = np.linalg.eigh(A @ A.T)
        # Löwdin-orthonormalized hybrid coefficients
        C = (V * w**-0.5) @ V.T @ A
        orbs = [dict(label=f'{SET_LABEL[oset].split()[0]} #{k + 1}', comps=dict(zip(basis, C[k])), hybrid=True)
                for k in range(len(lobes))]
        orbs += [dict(label=b, comps={b: 1.0}, hybrid=False) for b in rest]
        return orbs, C
    return [dict(label=PRETTY.get(b, b), comps={
                 b: 1.0}, hybrid=False) for b in PURE[oset]], None


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def aim_matrix(u, twist_deg=0.0):
    """Rotation taking local +x onto u, followed by a twist about that axis."""
    u = _unit(u)
    az, el = math.atan2(u[1], u[0]), math.asin(float(np.clip(u[2], -1, 1)))
    return _rz(az) @ _ry(-el) @ _rx(math.radians(twist_deg))


def _A(el, orb, parent=None, L=0.0, d=(1, 0, 0),
       aim='parent', twist=0.0, pos=None):
    """Atom spec. d: vector, 'lobeK' (K-th lobe of parent's set) or 'continue'."""
    return dict(el=el, orb=orb, parent=parent, L=L,
                d=d, aim=aim, twist=twist, pos=pos)


_hf_chain = [_A('F', 'sp3', aim=(1, 0, 0))]
for _k in range(4):
    if _k:
        _hf_chain.append(_A('F', 'sp3', len(_hf_chain) - 1, 1.80, 'continue'))
    _hf_chain.append(_A('H', 's', len(_hf_chain) -
                        1, 0.92, 'lobe0' if _k == 0 else 'lobe1'))

_benzene = []
for _k in range(6):
    _u = (math.cos(_k * PI / 3), math.sin(_k * PI / 3), 0)
    _benzene.append(_A('C', 'sp2', aim=_u, pos=1.40 * np.array(_u)))
for _k in range(6):
    _benzene.append(_A('H', 's', _k, 1.09, 'lobe0'))

PRESETS = {
    'Orbital gallery — d clovers and dz² torus': [_A('S', 'd', aim=(1, 0, 0))],
    'Orbital gallery — s sphere': [_A('H', 's', aim=(1, 0, 0))],
    'Orbital gallery — p dumbbells': [_A('C', 'p', aim=(1, 0, 0))],
    'Orbital gallery — f lobes': [_A('S', 'f', aim=(1, 0, 0))],
    'Benzene C₆H₆ — delocalized π / resonance': _benzene,
    'H₂ — s+s σ bond': [_A('H', 's', aim=(1, 0, 0)), _A('H', 's', 0, 0.74)],
    'N₂ — p+p: σ + 2π (triple bond)': [_A('N', 'p', aim=(1, 0, 0)), _A('N', 'p', 0, 1.10)],
    'Ethylene C₂H₄ — sp² σ + π': [
        _A('C', 'sp2', aim=(1, 0, 0)), _A('C', 'sp2', 0, 1.34, 'lobe0'),
        _A('H', 's', 0, 1.09, 'lobe1'), _A('H', 's', 0, 1.09, 'lobe2'),
        _A('H', 's', 1, 1.09, 'lobe1'), _A('H', 's', 1, 1.09, 'lobe2')],
    'Acetylene C₂H₂ — sp σ + 2π': [
        _A('C', 'sp', aim=(1, 0, 0)), _A('C', 'sp', 0, 1.20, 'lobe0'),
        _A('H', 's', 0, 1.06, 'lobe1'), _A('H', 's', 1, 1.06, 'lobe1')],
    'Water dimer — sp³ O–H···O hydrogen bond': [
        _A('O', 'sp3', aim=(1, 0, 0)), _A('H', 's', 0,
                                          0.96, 'lobe0'), _A('H', 's', 0, 0.96, 'lobe1'),
        _A('O', 'sp3', 1, 1.90, 'continue'), _A('H', 's', 3, 0.96, 'lobe1'), _A('H', 's', 3, 0.96, 'lobe2')],
    'HF dimer — F–H···F hydrogen bond': _hf_chain[:4],
    '(HF)₄ chain — 4 molecules, 3 hydrogen bonds': _hf_chain,
    '4 × H₂ — four interacting molecules': [
        _A('H', 's', aim=(1, 0, 0)), _A('H', 's', 0, 0.74),
        _A('H', 's', 1, 2.40, (0, 1, 0)), _A('H', 's', 2, 0.74, (-1, 0, 0)),
        _A('H', 's', 0, 2.40, (0, 0, 1)), _A('H', 's', 4, 0.74, (1, 0, 0)),
        _A('H', 's', 5, 2.40, (0, 1, 0)), _A('H', 's', 6, 0.74, (-1, 0, 0))],
    'PF₅ — sp³d (trigonal bipyramid)': [_A('P', 'sp3d', aim=(1, 0, 0))] +
    [_A('F', 'p', 0, 1.56, f'lobe{k}') for k in range(5)],
    'SF₆ — sp³d² (octahedral)': [_A('S', 'sp3d2', aim=(1, 0, 0))] +
    [_A('F', 'p', 0, 1.56, f'lobe{k}') for k in range(6)],
    'S(spdf cube) + 6 F — sp³d³f hybrids (hypothetical)': [_A('S', 'sp3d3f', aim=(1, 0, 0))] +
    [_A('F', 'p', 0, 1.65, f'lobe{k}') for k in range(6)],
    'd + d — illustrative σ / π / δ overlaps': [_A('S', 'd', aim=(1, 0, 0)), _A('S', 'd', 0, 2.25)],
    'f + f — illustrative σ / π / δ / φ symmetry': [_A('S', 'f', aim=(1, 0, 0)), _A('S', 'f', 0, 1.90)],
    'Custom pair (edit the atoms below)': [_A('C', 'sp2', aim=(1, 0, 0)), _A('C', 'sp2', 0, 1.50, 'lobe0')],
    'Custom 4-centre square (edit the atoms below)': [
        _A('N', 'p', aim=(1, 0, 0)), _A('N', 'p', 0, 1.8, (1, 0, 0)),
        _A('N', 'p', 1, 1.8, (0, 1, 0)), _A('N', 'p', 2, 1.8, (-1, 0, 0))],
}


def place_atoms(specs, scale=1.0, twists=None, overrides=None):
    twists, overrides = twists or {}, overrides or {}
    atoms = []
    for i, s in enumerate(specs):
        s = {**s, **overrides.get(i, {})}
        if s['parent'] is None:
            pos = np.zeros(3) if s['pos'] is None else np.asarray(
                s['pos'], float)
        else:
            p = atoms[s['parent']]
            d = s['d']
            if isinstance(d, str) and d.startswith('lobe'):
                # geometry follows the preset's set
                lobes = set_lobes(specs[s['parent']]['orb'])
                u = p['M0'] @ lobes[int(d[4:]) % len(lobes)]
            elif d == 'continue':
                gp = p['parent']
                u = p['pos'] - \
                    atoms[gp]['pos'] if gp is not None else np.array([
                                                                     1.0, 0, 0])
            else:
                u = np.asarray(d, float)
            pos = p['pos'] + s['L'] * scale * _unit(u)
        if s['aim'] == 'parent' and s['parent'] is not None:
            aim = atoms[s['parent']]['pos'] - pos
        else:
            aim = np.asarray(
                s['aim'] if s['aim'] != 'parent' else (
                    1, 0, 0), float)
        R = aim_matrix(aim, s['twist'] + twists.get(i, 0.0))
        orbs, C = build_orbitals(s['orb'])
        atoms.append(dict(idx=i, el=s['el'], orb=s['orb'], parent=s['parent'], pos=pos,
                          M=R @ (QFRAME.get(s['orb'], np.eye(3))
                                 if len(specs) > 1 else np.eye(3)),
                          M0=aim_matrix(
                              aim, specs[i]['twist']) @ QFRAME.get(specs[i]['orb'], np.eye(3)),
                          orbs=orbs, C=C, name=f"{s['el']}{i + 1}"))
    return atoms


def eval_orbital(atom, orb, P):
    """Evaluate one atomic/hybrid orbital at points P (N, 3) in Å."""
    q = (P - atom['pos']) @ atom['M']
    r = np.linalg.norm(q, axis=-1)
    rs = np.maximum(r, 1e-9)
    X, Y, Z = q[..., 0] / rs, q[..., 1] / rs, q[..., 2] / rs
    shells = ELEMENTS[atom['el']]['shells']
    out = np.zeros(r.shape)
    radial = {}
    for name, coef in orb['comps'].items():
        sh = name[0]
        if sh not in radial:
            n, zb = shells[sh]
            z = zb / BOHR
            radial[sh] = (2 * z) ** (n + 0.5) / \
                SQ(math.factorial(2 * n)) * r ** (n - 1) * np.exp(-z * r)
        out += coef * radial[sh] * YLM[name](X, Y, Z)
    return out


def orbital_alpha(atom, orb):
    v = ELEMENTS[atom['el']]['vsip']
    w = {k: c * c for k, c in orb['comps'].items()}
    return sum(v[k[0]] * c for k, c in w.items()) / sum(w.values())


def orbital_character(orb):
    w = {}
    for k, c in orb['comps'].items():
        w[k[0]] = w.get(k[0], 0) + c * c
    t = sum(w.values())
    return ' '.join(f"{k}{100 * v / t:.0f}%" for k,
                    v in w.items() if v / t > 0.005)


def make_grid(atoms, pad=3.0, target=180_000):
    P = np.array([a['pos'] for a in atoms])
    lo, hi = P.min(0) - pad, P.max(0) + pad
    h = max(0.1, (np.prod(hi - lo) / target) ** (1 / 3))
    axes = [np.arange(lo[k], hi[k] + h, h) for k in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing='ij'), -1)
    return dict(axes=axes, shape=G.shape[:3], h=h, lo=lo, pts=G.reshape(-1, 3))


def _phase_m(atom_a, oa, atom_b, ob):
    """Dominant angular momentum projection |m| of the overlap about the A→B axis."""
    pa, pb = atom_a['pos'], atom_b['pos']
    ax = pb - pa
    d = np.linalg.norm(ax)
    ez = ax / d
    tmp = np.array([1.0, 0, 0]) if abs(ez[0]) < 0.9 else np.array([0, 1.0, 0])
    ex = _unit(np.cross(ez, tmp))
    ey = np.cross(ez, ex)
    phis = np.linspace(0, 2 * PI, 16, endpoint=False)
    w = np.zeros(4)
    for t in np.linspace(-0.3, 1.3, 9):
        for rho in (0.25, 0.5, 0.8, 1.2):
            pts = pa + t * ax + rho * \
                (np.cos(phis)[:, None] * ex + np.sin(phis)[:, None] * ey)
            Fa = np.fft.fft(eval_orbital(atom_a, oa, pts))
            Fb = np.fft.fft(eval_orbital(atom_b, ob, pts))
            for m in range(4):
                c = (Fa[m] * np.conj(Fb[m])).real
                if m:
                    c += (Fa[-m] * np.conj(Fb[-m])).real
                w[m] += abs(c) * rho
    return int(np.argmax(w)), w


BOND_NAMES = ['σ', 'π', 'δ', 'φ']


def pi_networks(atoms, cands, used, cap, Psi, S, pos_of):
    """Find shared, one-electron p networks after the sigma skeleton is assigned."""
    eligible = {
        (a['idx'], k)
        for a in atoms if a['orb'] in ('sp', 'sp2') and cap[a['idx']] == 1
        for k, o in enumerate(a['orbs'])
        if not o['hybrid'] and set(o['comps']).issubset({'px', 'py', 'pz'})
        and (a['idx'], k) not in used
    }
    edges = [c for c in cands if c['m'] == 1
             and (c['a'], c['ka']) in eligible and (c['b'], c['kb']) in eligible]
    adj = {}
    for c in edges:
        a, b = (c['a'], c['ka']), (c['b'], c['kb'])
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    systems, visited = [], set()
    for start in sorted(adj):
        if start in visited:
            continue
        todo, component = [start], set()
        while todo:
            node = todo.pop()
            if node in component:
                continue
            component.add(node)
            todo.extend(adj[node] - component)
        visited |= component
        if len(component) < 3 or len(
                {a for a, _ in component}) != len(component):
            continue
        nodes = sorted(component)
        indices = [pos_of[node] for node in nodes]
        metric = S[np.ix_(indices, indices)]
        alpha = np.array(
            [orbital_alpha(atoms[a], atoms[a]['orbs'][k]) for a, k in nodes])
        H = 1.75 * (alpha[:, None] + alpha[None, :]) * metric / 2
        np.fill_diagonal(H, alpha)
        energies, coeff = eigh(H, metric)
        occ = np.clip(len(nodes) - 2 * np.arange(len(nodes)), 0, 2)
        fields = coeff.T @ Psi[indices]
        systems.append(dict(nodes=nodes, energies=energies, coeff=coeff,
                            occupancy=occ, fields=fields,
                            edges=[c for c in edges if (c['a'], c['ka']) in component
                                   and (c['b'], c['kb']) in component]))
        used.update(nodes)
        for a, _ in nodes:
            cap[a] -= 1
    return systems


def capacity(atom):
    v = ELEMENTS[atom['el']]['val']
    if atom['orb'] in ('p', 'd', 'f') and atom['el'] != 'H':
        v -= 2  # ns² pair is not in the selected (pure p/d/f) set
    n = len(atom['orbs'])
    return max(0, v if v <= n else 2 * n - v), v


def analyze(atoms, threshold=0.10):
    grid = make_grid(atoms)
    pts, h3 = grid['pts'], grid['h'] ** 3
    index, rows = [], []
    for a in atoms:
        for k, o in enumerate(a['orbs']):
            psi = eval_orbital(a, o, pts)
            rows.append(
                (psi / SQ(max((psi * psi).sum() * h3, 1e-30))).astype(np.float32))
            index.append((a['idx'], k))
    Psi = np.array(rows)
    S = (Psi @ Psi.T).astype(float) * h3
    pos_of = {key: i for i, key in enumerate(index)}

    cands = []
    for i, a in enumerate(atoms):
        for b in atoms[i + 1:]:
            d = np.linalg.norm(a['pos'] - b['pos'])
            if d > 1.5 * (ELEMENTS[a['el']]['rcov'] +
                          ELEMENTS[b['el']]['rcov']):
                continue
            for ka, oa in enumerate(a['orbs']):
                for kb, ob in enumerate(b['orbs']):
                    s = S[pos_of[(a['idx'], ka)], pos_of[(b['idx'], kb)]]
                    if abs(s) > threshold:
                        m, _ = _phase_m(a, oa, b, ob)
                        if m > 0 and (oa['hybrid'] or ob['hybrid']):
                            continue  # directional hybrids only make σ bonds; π comes from leftover p/d/f
                        cands.append(
                            dict(
                                a=a['idx'],
                                ka=ka,
                                b=b['idx'],
                                kb=kb,
                                S=float(s),
                                m=min(
                                    m,
                                    3),
                                d=d))
    cap = {a['idx']: capacity(a)[0] for a in atoms}
    used, bonds, sig_pairs, pi_systems = set(), [], set(), []
    for m in range(4):
        if m == 1:
            pi_systems = pi_networks(atoms, cands, used, cap, Psi, S, pos_of)
        for c in sorted([c for c in cands if c['m'] == m],
                        key=lambda c: -abs(c['S'])):
            pair = (c['a'], c['b'])
            if (c['a'], c['ka']) in used or (c['b'], c['kb']) in used:
                continue
            if cap[c['a']] < 1 or cap[c['b']] < 1:
                continue
            if m == 0 and pair in sig_pairs:
                continue  # at most one σ per atom pair
            used |= {(c['a'], c['ka']), (c['b'], c['kb'])}
            cap[c['a']] -= 1
            cap[c['b']] -= 1
            if m == 0:
                sig_pairs.add(pair)
            bonds.append(c)

    lone, warnings = [], []
    for a in atoms:
        nb = sum(1 for b in bonds if a['idx'] in (b['a'], b['b']))
        pi_e = sum(
            a['idx'] == i for system in pi_systems for i,
            _ in system['nodes'])
        e = capacity(a)[1] - nb - pi_e
        free = sorted([k for k in range(len(a['orbs'])) if (a['idx'], k) not in used],
                      key=lambda k: not a['orbs'][k]['hybrid'])
        pairs = min(max(0, e - len(free)), len(free))
        singles = max(0, min(e - 2 * pairs, len(free) - pairs))
        if e - 2 * pairs - singles > 0:
            warnings.append(f"{a['name']}: {e - 2 * pairs - singles} electron(s) do not fit its {a['orb']} set "
                            "(expected for hypervalent/teaching sets — pure sets omit the ns² pair).")
        for j, k in enumerate(free[:pairs + singles]):
            lone.append(dict(atom=a['idx'], k=k, n_e=2 if j < pairs else 1))
        if e < 0:
            warnings.append(
                f"{a['name']} formed more bonds than valence electrons allow.")

    hbonds = []
    sig_of = {}
    for b in bonds:
        if b['m'] == 0:
            sig_of.setdefault(b['a'], []).append((b['b'], b['ka']))
            sig_of.setdefault(b['b'], []).append((b['a'], b['kb']))
    for hA in atoms:
        if hA['el'] != 'H':
            continue
        donors = [(j, k) for j, k in sig_of.get(
            hA['idx'], []) if atoms[j]['el'] in HB_ATOMS]
        if not donors:
            continue
        D, kh = atoms[donors[0][0]], donors[0][1]
        for X in atoms:
            if X['el'] not in HB_ATOMS or X['idx'] == D['idx']:
                continue
            lps = [l for l in lone if l['atom'] == X['idx'] and l['n_e'] == 2]
            dist = np.linalg.norm(X['pos'] - hA['pos'])
            if not lps or not 1.4 <= dist <= 2.6:
                continue
            v1, v2 = D['pos'] - hA['pos'], X['pos'] - hA['pos']
            ang = math.degrees(
                math.acos(
                    np.clip(
                        v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1)))
            if ang <= 120:
                continue
            best = max(lps, key=lambda l: abs(
                S[pos_of[(X['idx'], l['k'])], pos_of[(hA['idx'], kh)]]))
            s = S[pos_of[(X['idx'], best['k'])], pos_of[(hA['idx'], kh)]]
            hbonds.append(dict(h=hA['idx'], kh=kh, donor=D['idx'], acc=X['idx'], kx=best['k'], d=dist,
                               angle=ang, S=float(s),
                               strength='strong' if dist < 1.8 else 'moderate' if dist < 2.2 else 'weak'))

    for b in bonds:
        A, B = atoms[b['a']], atoms[b['b']]
        aa, ab = orbital_alpha(A, A['orbs'][b['ka']]), orbital_alpha(
            B, B['orbs'][b['kb']])
        beta = 1.75 * b['S'] * (aa + ab) / 2
        energies, coeff = eigh(np.array([[aa, beta], [beta, ab]]),
                               np.array([[1.0, b['S']], [b['S'], 1.0]]))
        coeff *= np.where(coeff[0] < 0, -1, 1)
        b.update(
            alpha_a=aa,
            alpha_b=ab,
            E_bond=energies[0],
            E_anti=energies[1],
            coeff=coeff)
    for l in lone:
        l['alpha'] = orbital_alpha(
            atoms[l['atom']], atoms[l['atom']]['orbs'][l['k']])
    return dict(grid=grid, Psi=Psi, S=S, pos_of=pos_of, bonds=bonds, lone=lone, hbonds=hbonds,
                pi_systems=pi_systems,
                warnings=warnings)


def mo_field(res, b, sign=+1):
    pa, pb = res['pos_of'][(b['a'], b['ka'])], res['pos_of'][(b['b'], b['kb'])]
    c = b['coeff'][:, 0 if sign == 1 else 1]
    return c[0] * res['Psi'][pa] + c[1] * res['Psi'][pb]


PHASE = {0: ('#1f77b4', '#ff7f0e'), 1: ('#2ca02c', '#d62728'), 2: ('#9467bd', '#e377c2'),
         3: ('#8c564b', '#7f7f7f'), 'lp': ('#17becf', '#bcbd22'), 'ao': ('#4c78a8', '#e45756'),
         'hb': ('#ffd700', '#8a2be2')}
AO_PALETTE = [('#4c78a8', '#e45756'), ('#54a24b', '#f58518'), ('#b279a2', '#72b7b2'),
              ('#9d755d', '#eeca3b'), ('#2ca02c',
                                       '#d62728'), ('#17becf', '#ff9da6'),
              ('#1f77b4', '#ff7f0e'), ('#8c564b', '#bab0ac')]


def iso_meshes(field, grid, level, colors, name, opacity, step=1, group=None):
    F = field.reshape(grid['shape'])
    if not 0 < level < 1:
        raise ValueError(
            'The lobe surface fraction must lie between zero and one.')
    level = level * float(np.max(np.abs(F)))
    out = []
    for sgn, col in zip((+1, -1), colors):
        if not (sgn * F).min() < level < (sgn * F).max():
            continue
        v, f, _, _ = measure.marching_cubes(
            sgn * F, level, spacing=(grid['h'],) * 3, step_size=step)
        v = np.round(v + grid['lo'], 3)
        out.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                             color=col, opacity=opacity, name=f"{name} ({
            '+' if sgn > 0 else '−'})",
            legendgroup=group or name, showlegend=sgn > 0, hoverinfo='name', flatshading=False,
            lighting=dict(ambient=0.45, diffuse=0.8, specular=0.3, roughness=0.5)))
    return out


def _olabel(atoms, i, k):
    return f"{atoms[i]['name']} {atoms[i]['orbs'][k]['label']}"


def orbital_gallery(atoms, res, iso, opacity, focus):
    orbs = atoms[focus]['orbs']
    cols = min(3, len(orbs))
    rows = math.ceil(len(orbs) / cols)
    specs = [[{'type': 'scene'} if r * cols + c < len(orbs) else None
              for c in range(cols)] for r in range(rows)]
    fig = make_subplots(rows, cols, specs=specs,
                        subplot_titles=[o['label'] for o in orbs])
    center = atoms[focus]['pos']
    for k, o in enumerate(orbs):
        r, c = divmod(k, cols)
        meshes = iso_meshes(res['Psi'][res['pos_of'][(focus, k)]], res['grid'],
                            iso, PHASE['ao'], _olabel(atoms, focus, k), opacity)
        for mesh in meshes:
            mesh.showlegend = False
            fig.add_trace(mesh, r + 1, c + 1)
        fig.add_trace(go.Scatter3d(x=[center[0]], y=[center[1]], z=[center[2]],
                                   mode='markers', marker=dict(size=4, color='black'),
                                   showlegend=False, hoverinfo='skip'), r + 1, c + 1)
    fig.update_scenes(
        aspectmode='data',
        xaxis_title='x (Å)',
        yaxis_title='y (Å)',
        zaxis_title='z (Å)')
    fig.update_layout(
        height=max(
            420,
            340 * rows),
        margin=dict(
            l=0,
            r=0,
            t=40,
            b=0))
    return fig


def build_figure(atoms, res, mode, iso, opacity, focus=0,
                 orbital=None, bond=None, pi_mo=0):
    if mode.startswith('Orbital gallery'):
        return orbital_gallery(atoms, res, iso, opacity, focus)
    fig = make_subplots(1, 2, specs=[[{'type': 'scene'}, {'type': 'xy'}]], column_widths=[0.68, 0.32],
                        subplot_titles=('3D orbitals', 'Orbital energy levels (eV)'))
    grid, Psi, pos_of = res['grid'], res['Psi'], res['pos_of']
    traces = []
    selected = list(enumerate(res['bonds'])) if bond is None else [
        (bond, res['bonds'][bond])]
    if mode.startswith('Atomic orbitals — all'):
        step = 2 if Psi.shape[0] > 12 else 1
        for (i, k), p in pos_of.items():
            traces += iso_meshes(Psi[p], grid, iso, PHASE['ao'], _olabel(atoms, i, k),
                                 opacity, step, group=atoms[i]['name'])
    elif mode.startswith('Atomic orbitals — focus'):
        for k in range(len(atoms[focus]['orbs'])):
            if orbital is not None and k != orbital:
                continue
            traces += iso_meshes(Psi[pos_of[(focus, k)]], grid, iso, PHASE['ao'],
                                 _olabel(atoms, focus, k), opacity)
    elif mode.startswith('Delocalized'):
        for n, system in enumerate(res['pi_systems']):
            if mode.startswith('Delocalized π density'):
                density = np.sum(
                    system['occupancy'][:, None] * system['fields'] ** 2, axis=0)
                traces += iso_meshes(density, grid, iso, ('#9467bd', '#9467bd'),
                                     f"π system {n + 1}: total occupied electron density", opacity)
            else:
                traces += iso_meshes(system['fields'][pi_mo], grid, iso, PHASE['ao'],
                                     f"π system {n + 1}, MO {pi_mo + 1}, "
                                     f"{system['occupancy'][pi_mo]} electron(s)", opacity)
    elif mode.startswith('Bond contributors'):
        for n, b in selected:
            phase = 1 if b['S'] >= 0 else -1
            for i, k, sign in ((b['a'], b['ka'], 1), (b['b'], b['kb'], phase)):
                traces += iso_meshes(sign * Psi[pos_of[(i, k)]], grid, iso, PHASE['ao'],
                                     _olabel(atoms, i, k), opacity, group=_olabel(atoms, i, k))
    elif mode.startswith('Bonding'):
        for n, b in selected:
            traces += iso_meshes(mo_field(res, b, +1), grid, iso, PHASE['ao'],
                                 f"{BOND_NAMES[b['m']]} {atoms[b['a']]['name']}–{atoms[b['b']]['name']} #{n + 1}",
                                 opacity)
        for l in res['lone'] if bond is None else []:
            traces += iso_meshes(Psi[pos_of[(l['atom'], l['k'])]], grid, iso, PHASE['lp'],
                                 f"lone {
                'pair' if l['n_e'] == 2 else 'e⁻'} {
                _olabel(
                    atoms, l['atom'], l['k'])}",
                opacity * 0.8)
        for hb in res['hbonds'] if bond is None else []:
            fld = Psi[pos_of[(hb['acc'], hb['kx'])]].astype(
                float) * Psi[pos_of[(hb['h'], hb['kh'])]]
            traces += iso_meshes(fld, grid, iso, PHASE['hb'],
                                 f"H-bond overlap {atoms[hb['h']]['name']}···{atoms[hb['acc']]['name']}",
                                 min(1.0, opacity + 0.2))
    elif mode.startswith('Antibonding'):
        for n, b in selected:
            traces += iso_meshes(mo_field(res, b, -1), grid, iso, PHASE['ao'],
                                 f"{BOND_NAMES[b['m']]}* {atoms[b['a']]['name']}–{atoms[b['b']]['name']} #{n + 1}",
                                 opacity)
    else:  # overlap density
        for n, b in selected:
            fld = np.sign(b['S']) * Psi[pos_of[(b['a'], b['ka'])]
                                        ].astype(float) * Psi[pos_of[(b['b'], b['kb'])]]
            traces += iso_meshes(fld, grid, iso, ('#2ca02c', '#9467bd'),
                                 f"overlap ψaψb {BOND_NAMES[b['m']]} {atoms[b['a']]['name']}–{atoms[b['b']]['name']}",
                                 opacity)
        for hb in res['hbonds']:
            fld = Psi[pos_of[(hb['acc'], hb['kx'])]].astype(
                float) * Psi[pos_of[(hb['h'], hb['kh'])]]
            traces += iso_meshes(fld, grid, iso, PHASE['hb'],
                                 f"H-bond overlap {atoms[hb['h']]['name']}···{atoms[hb['acc']]['name']}", opacity)
    for t in traces:
        fig.add_trace(t, 1, 1)

    sticks = {}
    for b in res['bonds']:
        sticks.setdefault((b['a'], b['b']), []).append(b['m'])
    for (i, j), ms in sticks.items():
        P = np.array([atoms[i]['pos'], atoms[j]['pos']])
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='lines',
                                   line=dict(color='#666', width=4 + 3 * (len(ms) - 1)), showlegend=False,
                                   hoverinfo='text',
                                   text=f"{atoms[i]['name']}–{atoms[j]['name']}: " + ' + '.join(BOND_NAMES[m] for m in ms)),
                      1, 1)
    for hb in res['hbonds']:
        P = np.array([atoms[hb['h']]['pos'], atoms[hb['acc']]['pos']])
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='lines',
                                   line=dict(color='gold', width=5, dash='dash'), name='hydrogen bond',
                                   legendgroup='hb', showlegend=False), 1, 1)
    for system in res['pi_systems']:
        for edge in system['edges']:
            P = np.array([atoms[edge['a']]['pos'], atoms[edge['b']]['pos']])
            fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='lines',
                                       line=dict(
                color='#9467bd', width=7, dash='dot'),
                hovertext='Connected p orbitals in a shared π system',
                hoverinfo='text', showlegend=False), 1, 1)
    P = np.array([a['pos'] for a in atoms])
    fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='markers+text',
                               text=[a['name'] for a in atoms], textposition='top center',
                               marker=dict(size=[ELEMENTS[a['el']]['size'] for a in atoms],
                                           color=[ELEMENTS[a['el']]['color']
                                                  for a in atoms],
                                           line=dict(color='black', width=1)),
                               hovertext=[
                                   f"{a['name']} ({SET_LABEL[a['orb']]})" for a in atoms],
                               hoverinfo='text', name='atoms'), 1, 1)
    _energy_panel(fig, atoms, res, bond)
    fig.update_layout(height=700, margin=dict(l=0, r=10, t=40, b=0),
                      legend=dict(font=dict(size=10), itemsizing='constant'),
                      scene=dict(
        aspectmode='data',
        xaxis_title='x (Å)',
        yaxis_title='y (Å)',
        zaxis_title='z (Å)'),
        uirevision='orbital-camera')
    return fig


def _energy_panel(fig, atoms, res, bond=None):
    x0, tick_x, tick_t = 0, [], []
    for n, b in enumerate(res['bonds']):
        if bond is not None and n != bond:
            continue
        c = PHASE[b['m']][0]
        lv = [(x0, b['alpha_a'], 'AO'), (x0 + 1, b['E_bond'], 'b'), (x0 + 1, b['E_anti'], 'a'),
              (x0 + 2, b['alpha_b'], 'AO')]
        for x, E, kind in lv:
            fig.add_trace(go.Scatter(x=[x - 0.35, x + 0.35], y=[E, E], mode='lines', showlegend=False,
                                     line=dict(color=c if kind != 'AO' else '#888', width=3,
                                               dash='dash' if kind == 'a' else 'solid'),
                                     hovertext=f"{BOND_NAMES[b['m']]}{'*' if kind == 'a' else ''} {E:.2f} eV",
                                     hoverinfo='text'), 1, 2)
        for xa, Ea in ((x0 + 0.35, b['alpha_a']), (x0 + 1.65, b['alpha_b'])):
            for Em in (b['E_bond'], b['E_anti']):
                fig.add_trace(go.Scatter(x=[xa, x0 + 1 + (-0.35 if xa < x0 + 1 else 0.35)], y=[Ea, Em],
                                         mode='lines', line=dict(color='#bbb', width=1, dash='dot'),
                                         showlegend=False, hoverinfo='skip'), 1, 2)
        fig.add_trace(go.Scatter(x=[x0 + 0.9, x0 + 1.1], y=[b['E_bond']] * 2, mode='markers',
                                 marker=dict(color='black', size=6), showlegend=False, hoverinfo='skip'), 1, 2)
        tick_x.append(x0 + 1)
        tick_t.append(
            f"{BOND_NAMES[b['m']]} {atoms[b['a']]['name']}–{atoms[b['b']]['name']}")
        x0 += 3
    for n, system in enumerate(res['pi_systems']):
        for j, (E, occ) in enumerate(
                zip(system['energies'], system['occupancy'])):
            fig.add_trace(go.Scatter(x=[x0 - 0.35, x0 + 0.35], y=[E, E],
                                     mode='lines', line=dict(color='#9467bd', width=3),
                                     hovertext=f"π MO {
                j +
                1}: {occ} electron(s), {
                E:.2f} eV",
                hoverinfo='text', showlegend=False), 1, 2)
            if occ:
                dots = [x0 - 0.1, x0 + 0.1] if occ == 2 else [x0]
                fig.add_trace(go.Scatter(x=dots, y=[E] * len(dots), mode='markers',
                                         marker=dict(color='black', size=5),
                                         hoverinfo='skip', showlegend=False), 1, 2)
        tick_x.append(x0)
        tick_t.append(f"shared π system {n + 1}")
        x0 += 3
    if res['lone']:
        for j, l in enumerate(res['lone']):
            fig.add_trace(go.Scatter(x=[x0 - 0.35, x0 + 0.35], y=[l['alpha']] * 2, mode='lines',
                                     line=dict(color=PHASE['lp'][0], width=3), showlegend=False,
                                     hovertext=f"lone {
                _olabel(
                    atoms, l['atom'], l['k'])} {
                l['alpha']:.2f} eV",
                hoverinfo='text'), 1, 2)
            dots = [x0 - 0.1, x0 + 0.1] if l['n_e'] == 2 else [x0]
            fig.add_trace(go.Scatter(x=dots, y=[l['alpha']] * len(dots), mode='markers',
                                     marker=dict(color='black', size=5), showlegend=False, hoverinfo='skip'), 1, 2)
        tick_x.append(x0)
        tick_t.append('lone pairs')
    fig.update_xaxes(
        tickvals=tick_x,
        ticktext=tick_t,
        tickangle=-60,
        tickfont=dict(
            size=9),
        row=1,
        col=2)
    fig.update_yaxes(title_text='E (eV)', row=1, col=2)


def summary_html(atoms, res, mode='', focus=0,
                 orbital=None, bond=None, pi_mo=0):
    e = _html.escape
    rows = ''.join(
        f"<tr><td>{BOND_NAMES[b['m']]}</td><td>{e(_olabel(atoms, b['a'], b['ka']))} + "
        f"{e(_olabel(atoms,
                     b['b'],
                     b['kb']))}</td><td>{b['d']:.2f}</td><td>{b['S']:+.3f}</td>"
        f"<td>{b['E_bond']:.2f}</td><td>{b['E_anti']:.2f}</td></tr>" for b in res['bonds'])
    counts = {m: sum(1 for b in res['bonds'] if b['m'] == m) for m in range(4)}
    head = ', '.join(
        f"{v} {
            BOND_NAMES[m]}" for m,
        v in counts.items() if v) or 'no covalent bonds'
    out = (f"<b>Bonds:</b> {head}; <b>{len(res['hbonds'])}</b> hydrogen bond(s)."
           "<table style='font-size:12px'><tr><th>type</th><th>orbitals</th><th>d (Å)</th><th>S</th>"
           f"<th>E bonding (eV)</th><th>E antibonding (eV)</th></tr>{rows}</table>")
    explanation = ("<b>Qualitative teaching model, not a quantum-chemistry bond prediction.</b> "
                   "d/f and spd/spdf examples illustrate angular symmetry; they do not establish "
                   "d/f bonding in sulfur, carbon, PF₅ or SF₆.<br>"
                   "<b>Shape key:</b> s = sphere; p = two lobes; dxy, dxz, dyz and dx²−y² = "
                   "four-lobed clovers; dz² = two lobes + an equatorial torus. Hybrids mix these "
                   "shapes, so an spd hybrid is not a pure d clover. Blue/orange mean opposite "
                   "wavefunction phases, not electron charge.<br>"
                   "<b>Bond key:</b> σ = head-on overlap along the internuclear axis; "
                   "π = side-on overlap of parallel p orbitals, above and below that axis. "
                   "A π bond does not arise from σ bonds resonating. Resonance describes "
                   "electrons delocalized over three or more connected p orbitals; it is not "
                   "bonds switching back and forth.<br>")
    if mode.startswith('Atomic orbitals') or mode.startswith(
            'Orbital gallery'):
        labels = [o['label'] for k, o in enumerate(atoms[focus]['orbs'])
                  if orbital is None or k == orbital]
        explanation += f"<b>Focus:</b> {e(atoms[focus]['name'])}: {e(', '.join(labels))}.<br>"
    elif bond is not None:
        b = res['bonds'][bond]
        explanation += (f"<b>Selected {BOND_NAMES[b['m']]} overlap:</b> "
                        f"{e(_olabel(atoms, b['a'], b['ka']))} + "
                        f"{e(_olabel(atoms, b['b'], b['kb']))}. "
                        "Use Bond contributors → Bonding MO → Antibonding MO to compare.<br>")
    for n, system in enumerate(res['pi_systems']):
        labels = ', '.join(_olabel(atoms, i, k) for i, k in system['nodes'])
        explanation += (f"<b>Shared π system {n + 1}:</b> {e(labels)}; "
                        f"{sum(system['occupancy'])} π electrons in "
                        f"{sum(system['occupancy'] > 0)} occupied delocalized MOs. "
                        "Purple dotted contacts mark the connected p-orbital network, not extra localized bonds.<br>")
        if mode.startswith('Delocalized π density'):
            explanation += ("<b>Displayed:</b> summed occupied π electron density "
                            "Σ nᵢ|ψᵢ|² (purple, nonnegative), not a single orbital's phase.<br>")
        elif mode.startswith('Delocalized'):
            weights = ', '.join(f"{_olabel(atoms, i, k)}: {c:+.3f}"
                                for (i, k), c in zip(system['nodes'], system['coeff'][:, pi_mo]))
            explanation += (f"<b>Displayed π MO {pi_mo + 1}:</b> "
                            f"{system['occupancy'][pi_mo]} electron(s); "
                            f"coefficients {e(weights)}.<br>")
    out = explanation + out
    if res['hbonds']:
        out += "<b>Hydrogen bonds</b><table style='font-size:12px'><tr><th>D–H···A</th><th>H···A (Å)</th>" \
               "<th>angle</th><th>S(lp,H)</th><th>strength</th></tr>" + ''.join(
                   f"<tr><td>{atoms[h['donor']]['name']}–{atoms[h['h']]['name']}···{atoms[h['acc']]['name']}</td>"
                   f"<td>{h['d']:.2f}</td><td>{h['angle']:.0f}°</td><td>{h['S']:+.3f}</td><td>{h['strength']}</td></tr>"
                   for h in res['hbonds']) + '</table>'
    if res['lone']:
        out += '<b>Non-bonding:</b> ' + ', '.join(
            f"{e(_olabel(atoms, l['atom'], l['k']))} ({'pair' if l['n_e'] == 2 else '1 e⁻'})" for l in res['lone'])
    out += '<br><b>Orbital make-up:</b> ' + '; '.join(
        f"{a['name']} {SET_LABEL[a['orb']]}: " + ', '.join(
            f"{e(o['label'])} [{orbital_character(o)}]" for o in a['orbs'] if o['hybrid'])
        for a in atoms if a['C'] is not None)
    for w in res['warnings']:
        out += f"<br><span style='color:#b35c00'>⚠ {e(w)}</span>"
    return out


MODES = ['Orbital gallery — separate lobes', 'Atomic orbitals — focus atom',
         'Bond contributors — selected atomic pair', 'Bonding MOs + lone pairs + H-bonds',
         'Antibonding MOs (σ*, π*, …)', 'Delocalized π density — all occupied MOs',
         'Delocalized π — selected molecular orbital',
         'Overlap density ψa·ψb (where bonds form)', 'Atomic orbitals — all atoms',
         ]
_style = dict(description_width='130px')
_lay = widgets.Layout(width='380px')
w_preset = widgets.Dropdown(options=list(PRESETS), value='Orbital gallery — d clovers and dz² torus',
                            description='System', style=_style, layout=_lay)
w_mode = widgets.Dropdown(
    options=MODES,
    description='Show',
    style=_style,
    layout=_lay)
w_focus = widgets.Dropdown(description='Focus atom', style=_style, layout=_lay)
w_el = widgets.Dropdown(
    options=list(ELEMENTS),
    description='  element',
    style=_style,
    layout=_lay)
w_set = widgets.Dropdown(options=[(SET_LABEL[o], o) for o in ORBITAL_SETS], description='  orbital set',
                         style=_style, layout=_lay)
w_orbital = widgets.Dropdown(
    description='Individual AO',
    style=_style,
    layout=_lay)
w_bond = widgets.Dropdown(
    description='Bond / overlap',
    style=_style,
    layout=_lay)
w_pi = widgets.Dropdown(description='Shared π MO', style=_style, layout=_lay)
w_reset = widgets.Button(
    description='Reset atom edits',
    layout=widgets.Layout(
        width='160px'))
_slk = dict(continuous_update=False, style=_style, layout=_lay)
w_scale = widgets.FloatSlider(
    value=1.0,
    min=0.6,
    max=1.8,
    step=0.05,
    description='Distance ×',
    **_slk)
w_twist = widgets.FloatSlider(
    value=0,
    min=-180,
    max=180,
    step=15,
    description='Focus twist (°)',
    **_slk)
w_iso = widgets.FloatSlider(
    value=0.22,
    min=0.05,
    max=0.65,
    step=0.01,
    description='Lobe fraction',
    **_slk)
w_op = widgets.FloatSlider(
    value=0.80,
    min=0.15,
    max=1.0,
    step=0.05,
    description='Opacity',
    **_slk)
w_thr = widgets.FloatSlider(
    value=0.10,
    min=0.03,
    max=0.40,
    step=0.01,
    description='Overlap threshold',
    **_slk)
out_fig, out_txt = widgets.Output(), widgets.HTML()
_state = dict(overrides={}, twists={}, sync=False, result=None)


def _key(i):
    return (w_preset.value, i)


def _sync_focus(*_):
    specs = PRESETS[w_preset.value]
    _state['sync'] = True
    try:
        names = [f"{s['el']}{i + 1}" for i, s in enumerate(specs)]
        w_focus.options = [(n, i) for i, n in enumerate(names)]
        i = w_focus.value if w_focus.value is not None and w_focus.value < len(
            specs) else 0
        w_focus.value = i
        o = {**specs[i], **_state['overrides'].get(_key(i), {})}
        w_el.value, w_set.value = o['el'], o['orb']
        w_twist.value = _state['twists'].get(_key(i), 0.0)
    finally:
        _state['sync'] = False


def render(*_):
    if _state['sync']:
        return
    specs = PRESETS[w_preset.value]
    ov = {
        i: v for (
            p,
            i),
        v in _state['overrides'].items() if p == w_preset.value}
    tw = {
        i: v for (
            p,
            i),
        v in _state['twists'].items() if p == w_preset.value}
    with out_fig:
        out_fig.clear_output(wait=True)
        try:
            atoms = place_atoms(specs, w_scale.value, tw, ov)
            res = analyze(atoms, w_thr.value)
            _state['result'] = (atoms, res)
            _sync_choices(atoms, res)
            fig = build_figure(atoms, res, w_mode.value, w_iso.value, w_op.value, w_focus.value or 0,
                               w_orbital.value, w_bond.value, w_pi.value or 0)
            try:
                fig.show()
            except (ImportError, ValueError) as exc:
                display(HTML(f"<p>Plotly renderer unavailable: {_html.escape(str(exc))}. "
                             "Using embedded HTML instead.</p>"))
                display(
                    HTML(
                        fig.to_html(
                            include_plotlyjs=True,
                            full_html=False)))
            out_txt.value = summary_html(atoms, res, w_mode.value, w_focus.value or 0,
                                         w_orbital.value, w_bond.value, w_pi.value or 0)
        except Exception:
            out_txt.value = '<pre>' + \
                _html.escape(traceback.format_exc()) + '</pre>'


def _sync_choices(atoms, res):
    _state['sync'] = True
    try:
        focus = w_focus.value or 0
        key = (w_preset.value, focus, atoms[focus]['orb'],
               tuple((b['a'], b['ka'], b['b'], b['kb']) for b in res['bonds']))
        changed = key != _state.get('choice_key')
        _state['choice_key'] = key
        choices = [('All (superposed)', None)] + [(o['label'], k)
                                                  for k, o in enumerate(atoms[focus]['orbs'])]
        old = w_orbital.value
        w_orbital.options = choices
        w_orbital.value = old if not changed and old in [v for _, v in choices] else next(
            (k for k, o in enumerate(atoms[focus]['orbs']) if o['label'] in ('dxy', 'pz')), 0)
        old = w_bond.value
        choices = [('All bonds (busy)', None)] + [
            (f"{BOND_NAMES[b['m']]}: {_olabel(atoms, b['a'], b['ka'])} + "
             f"{_olabel(atoms, b['b'], b['kb'])}", k) for k, b in enumerate(res['bonds'])]
        w_bond.options = choices
        w_bond.value = old if not changed and old in [v for _, v in choices] else next(
            (k for k, b in enumerate(res['bonds']) if b['m'] == 1), 0 if res['bonds'] else None)
        n = min((len(s['energies']) for s in res['pi_systems']), default=0)
        old = w_pi.value
        w_pi.options = [(f"MO {k + 1} ({res['pi_systems'][0]['occupancy'][k]} e⁻)", k)
                        for k in range(n)] or [('No shared π system', None)]
        w_pi.value = old if old is not None and old < n else (0 if n else None)
        w_orbital.disabled = not w_mode.value.startswith(
            'Atomic orbitals — focus')
        w_bond.disabled = w_mode.value.startswith(
            ('Atomic', 'Orbital gallery', 'Delocalized'))
        w_pi.disabled = w_mode.value != 'Delocalized π — selected molecular orbital' or not n
        if w_mode.value.startswith('Delocalized') and not n:
            res['warnings'].append('No shared π network at this geometry/threshold. Try benzene; '
                                   'ethylene has a localized two-centre π bond.')
    finally:
        _state['sync'] = False


def _on_atom_edit(*_):
    if _state['sync']:
        return
    i = w_focus.value or 0
    _state['overrides'][_key(i)] = dict(el=w_el.value, orb=w_set.value)
    _state['twists'][_key(i)] = w_twist.value
    render()


def _on_focus(*_):
    if _state['sync']:
        return
    _sync_focus()
    render()


def _on_preset(*_):
    _sync_focus()
    _state['sync'] = True
    try:
        w_orbital.value = None
        w_bond.value = None
        w_mode.value = ('Orbital gallery — separate lobes' if len(PRESETS[w_preset.value]) == 1
                        else 'Delocalized π density — all occupied MOs' if w_preset.value.startswith('Benzene')
                        else 'Bond contributors — selected atomic pair')
    finally:
        _state['sync'] = False
    render()


def _on_reset(_):
    for k in [k for k in _state['overrides'] if k[0] == w_preset.value]:
        del _state['overrides'][k]
    for k in [k for k in _state['twists'] if k[0] == w_preset.value]:
        del _state['twists'][k]
    _sync_focus()
    render()


w_preset.observe(_on_preset, 'value')
w_focus.observe(_on_focus, 'value')
for w in (w_el, w_set, w_twist):
    w.observe(_on_atom_edit, 'value')
for w in (w_mode, w_scale, w_iso, w_op, w_thr, w_orbital, w_bond, w_pi):
    w.observe(render, 'value')
w_reset.on_click(_on_reset)


def show_explorer():
    _sync_focus()
    display(widgets.VBox([
        widgets.HTML("<b>Start with the d gallery to see clovers.</b> For bonds, choose ethylene "
                     "and select its π pair; for resonance, choose benzene. "
                     "Blue/orange = opposite phase, not charge."),
        widgets.HBox([widgets.VBox([w_preset, w_mode, w_bond, w_pi, w_scale, w_iso, w_op, w_thr]),
                      widgets.VBox([w_focus, w_el, w_set, w_orbital, w_twist, w_reset])]),
        out_fig, out_txt]))
    render()

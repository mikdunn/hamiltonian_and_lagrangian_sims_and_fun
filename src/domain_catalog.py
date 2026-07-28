"""Metadata for all simulation folders in this repository."""

DOMAINS = {
    "01_celestial_mechanics": {
        "title": "Celestial mechanics and star-chart predictors",
        "data": "JPL ephemerides, Gaia, telescope astrometry",
        "cs": "Symplectic integration, inverse problems, chaos",
    },
    "02_molecular_dynamics": {
        "title": "Molecular dynamics simulations",
        "data": "PDB, NMR, SAXS, cryo-EM trajectories",
        "cs": "Sampling, differentiable simulation, ML force fields",
    },
    "03_enzyme_kinetics": {
        "title": "Enzyme kinetics for specific molecules",
        "data": "Product/substrate time courses, fluorescence, absorbance",
        "cs": "Differentiable ODE/PDE inference and model selection",
    },
    "04_protein_binding_energy": {
        "title": "Protein binding energy predictions",
        "data": "PDBbind, BindingDB, structural assays",
        "cs": "High-dimensional optimization and hybrid physics-ML",
    },
    "05_dna_repair": {
        "title": "DNA repair simulations",
        "data": "Microscopy foci, single-molecule tracks, damage-seq",
        "cs": "PDE-constrained inference and stochastic simulation",
    },
    "06_circuit_analysis": {
        "title": "Circuit analysis and EM fields",
        "data": "Voltage/current traces, S-parameters, impedance spectra",
        "cs": "Model reduction, control, network synthesis",
    },
    "07_computer_engineering_dynamics": {
        "title": "Computer engineering information-flow dynamics",
        "data": "Perf counters, thermal traces, network logs",
        "cs": "Complexity analogies, scheduling control, congestion theory",
    },
    "08_music_acoustics": {
        "title": "Music and acoustical signal dynamics",
        "data": "Recorded notes, impulse responses, MIDI",
        "cs": "Physics-based synthesis and structured generation",
    },
    "09_bond_energy_pathways": {
        "title": "Bond energies and reaction pathways",
        "data": "Spectroscopy, thermochemical tables, DFT outputs",
        "cs": "Potential-surface optimization and ML potentials",
    },
    "10_weather_dynamics": {
        "title": "Weather dynamics simulations",
        "data": "ERA5, GFS, satellite, radiosonde",
        "cs": "Large PDE solvers, assimilation, uncertainty quantification",
    },
    "11_fluid_dynamics": {
        "title": "Fluid dynamics (general)",
        "data": "PIV, flow videos, CFD datasets",
        "cs": "Structure-preserving discretization and turbulence ML",
    },
    "12_coupled_oscillators": {
        "title": "Coupled oscillators and mechanical field analogs",
        "data": "Motion capture, tracked video, sensor arrays",
        "cs": "Lattice models, dispersion, field-theory learning",
    },
    "13_strings_beams_membranes": {
        "title": "Vibrating strings, beams, and membranes",
        "data": "Audio, high-speed video, laser vibrometry",
        "cs": "Inverse PDE learning and modal synthesis",
    },
    "14_em_resonators": {
        "title": "Electromagnetic experiments and coupled resonators",
        "data": "S-parameters, field maps, time-domain responses",
        "cs": "Inverse design and structured EM model reduction",
    },
    "15_heat_diffusion_waves": {
        "title": "Heat conduction and diffusion waves",
        "data": "IR thermography, thermistor arrays",
        "cs": "Inverse heat conduction and thermal control",
    },
    "16_molecular_waves": {
        "title": "Molecular waves and spectra",
        "data": "IR/Raman/microwave spectra, pump-probe data",
        "cs": "Spectral inversion, quantum control, ML spectra",
    },
    "17_water_waves": {
        "title": "Water waves and free-surface dynamics",
        "data": "Wave tank video, buoy data, altimetry",
        "cs": "Free-boundary PDE inference and ocean assimilation",
    },
    "18_acoustical_dynamics": {
        "title": "Acoustical dynamics in rooms and media",
        "data": "Microphone arrays, impulse responses, ultrasonics",
        "cs": "Inverse acoustics and spatial audio simulation",
    },
    "19_seismic_elastic_waves": {
        "title": "Seismic and elastic waves",
        "data": "Seismometers, structural sensors, shake-table traces",
        "cs": "Seismology inverse problems and early warning ML",
    },
    "20_plasma_waves": {
        "title": "Plasma waves and collective modes",
        "data": "Langmuir probes, spectroscopy, space-plasma records",
        "cs": "Kinetic simulation and stability analysis",
    },
    "21_quantum_wave_mechanics": {
        "title": "Quantum wave mechanics",
        "data": "Spectra, scattering cross sections, cold-atom data",
        "cs": "Quantum simulation and state-learning methods",
    },
    "22_traffic_crowd_info_waves": {
        "title": "Traffic, crowd, and information-flow waves",
        "data": "Traffic sensors, GPS traces, crowd video, packet logs",
        "cs": "Network optimization and multi-agent control",
    },
}

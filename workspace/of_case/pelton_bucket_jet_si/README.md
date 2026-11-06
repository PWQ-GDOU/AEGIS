# Focused Pelton bucket SI baseline

This case reproduces the native OpenFOAM topology used by Step 30 with explicit SI units.

## Unit contract

- OpenFOAM coordinates: m
- `U`: m/s
- `p`: m²/s² (kinematic pressure); multiply by `rho=1000 kg/m³` for Pa
- `wallShearStress`: m²/s² (kinematic); multiply by `rho` for Pa
- Step 30 internal coordinates: mm (`coordinate_scale_to_mm=1000`)

The background vertices retain historical millimetre-valued literals and are converted exactly once by `blockMeshDict: scale 0.001`. The STL is converted independently with `surfaceTransformPoints -scale '(0.001 0.001 0.001)'`.

## Physical inputs

- Jet speed: 104.8 m/s
- Jet diameter: 0.24 m
- Water kinematic viscosity: 1e-6 m²/s
- Reynolds number: 2.5152e7
- Turbulence intensity: 5%
- Turbulence length scale: `0.07D = 0.0168 m`
- `k = 1.5 (I U)² = 41.1864 m²/s²`
- `omega = sqrt(k)/(Cmu^0.25 L) = 697.44025 s⁻¹`, with `Cmu=0.09`

The historical `omega=4.11864e6 s⁻¹` is not reused because it embeds the old length-scale error.

## Run

OpenFOAM v2412 commands must already be available in the shell:

```bash
export SOURCE_STL=/absolute/path/to/dec_full_runner_local.stl
./setup_and_run.sh
```

The script:

1. converts the STL from mm coordinates to m;
2. rebuilds and checks the mesh;
3. initializes `U` with `potentialFoam`;
4. runs 100 conservative SIMPLE iterations;
5. continues to at most 3000 iterations with stable relaxation;
6. writes wall shear stress, cell centres, and the native bucket VTP.

A completed run must still be inspected for residual convergence. High-Re separated steady RANS may reach a bounded statistical plateau without satisfying the strict `1e-4` residualControl threshold; this must be reported rather than hidden.

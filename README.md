# Coupled Free-Flow and Porous-Media Solver (FEniCS + PorePy + preCICE)

This repository couples two separate physics solvers using preCICE:

- `free_flow_participant`: free-flow side (FEniCS/DOLFIN), solving a Navier-Stokes-type problem.
- `porous_media_participant`: porous-medium side (PorePy), solving Darcy flow.

The two participants exchange interface data in a serial-implicit coupling:

- FreeFlow reads `Velocity` and writes `Force`.
- PorousMedia reads `Force` and writes `Velocity`.

The coupling configuration is in `precice-config.xml`.

## What the participant codes do

### `free_flow_participant/main.py`

This script:

1. Builds a 2D FEniCS mesh and mixed function space for velocity-pressure.
2. Initializes coupling through `fenicsprecice.Adapter` using `fenics-adapter-config.json`.
3. In each coupling iteration:
   - reads interface data from preCICE,
   - updates boundary conditions on the coupling boundary,
   - solves the nonlinear free-flow problem,
   - computes normal traction on the interface,
   - writes traction (`Force`) back to preCICE.
4. Exports final velocity and pressure to `output/`.

### `porous_media_participant/main.py`

This script:

1. Creates a PorePy Darcy model (`PorousMediaProblem` from `ppm_model.py`).
2. Initializes coupling through the local adapter (`porepyprecice/`) using `porepy-adapter-config.json`.
3. In each coupling iteration:
   - reads interface data from preCICE,
   - updates porous-medium boundary conditions,
   - solves the Darcy system,
   - computes boundary Darcy flux,
   - writes velocity/flux data to preCICE.
4. Finalizes coupling and exports pressure/flux output.

## Quick run guide with Docker

The project now runs inside a Docker container. The image sets up the FEniCS/DOLFIN stack for `free_flow_participant` and builds PorePy from source for `porous_media_participant`, so you do not need to manage two separate local Python environments.

### 1) Build the image

```bash
cd /home/trygve/PhD/Precice_Prosjekt/HPC_porepy_project
docker build -t hpc-porepy-coupled:latest .
```

### 2) Run the coupled case

The container entrypoint starts `porous_media_participant` first and then `free_flow_participant`.

```bash
docker run --rm hpc-porepy-coupled:latest
```

### 3) What the container includes

- Ubuntu 24.04 base image.
- Legacy FEniCS/DOLFIN from the archived FEniCS installation route.
- preCICE runtime package.
- PorePy installed from source.
- A run script at `docker/run_coupled.sh` that starts both participants.

### 4) Outputs

- Coupling data and exports are written to the existing `output/` and participant-specific preCICE export directories.
- The scripts still use the same adapter JSON files and `precice-config.xml` inside the container.
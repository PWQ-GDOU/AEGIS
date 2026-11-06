# Authoritative geometry

`single_bucket_target_authoritative_local_m.stl` is the complete target bucket exported from COMSOL 6.4 after importing the authoritative full-runner Parasolid.

- Full-runner Parasolid SHA256: `a597476082e104cfa1b70c48a06adb46629e60a0cfd0175399471ecd2fcb6313`
- Selected repeated CAD solid: `imp1(12)`
- Local metre-coordinate STL SHA256: `181eb659091f187e1286ca27b689d654a83b5522b28d6f39aceaa4ebc3fc38ca`
- Coordinate units: m
- Approximate bounds: x `0.043468..0.421344`, y `-0.187982..0.172085`, z `-0.406253..0.402147` m

The geometry is already in metres. Do not apply an additional `0.001` scale. `setup_and_run.sh` verifies this exact hash before rebuilding the mesh.

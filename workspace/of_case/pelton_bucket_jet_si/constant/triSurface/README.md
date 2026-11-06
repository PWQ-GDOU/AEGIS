# Geometry input

The large STL is intentionally excluded from Git. Set `SOURCE_STL` to the historical millimetre-coordinate `dec_full_runner_local.stl` before running `setup_and_run.sh`. The script creates `dec_full_runner_local_si.stl` with a `0.001` coordinate scale.

Expected SI bounds are approximately:

- x: 0.0433 to 1.3457 m
- y: -0.1727 to 0.1723 m
- z: -0.4042 to 0.4002 m

Validated SI STL SHA256 from the production run:

`a27be332b508d1ee7c7705bbdbb18547360a78c618b61282f2cc4560040d3f3e`

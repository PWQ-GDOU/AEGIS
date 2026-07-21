"""
Step 11: 生成 OpenFOAM 仿真 Case

基于DEC文献(梁权伟/肖业祥2021&2024)的完整CFD设置
直接生成所有配置文件，复制到WSL即可运行

Case结构:
  pelton_jet/          → 简化2D/3D射流+喷嘴侵蚀验证
  pelton_bucket/       → 完整3D水斗旋转仿真 (待后续)
"""
import os
from pathlib import Path
from textwrap import dedent

OUTPUT = Path(r"B:\undertaking\学业\非专业相关\比赛\挑战杯\水轮机\workspace\of_case\pelton_jet")
OUTPUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Create all subdirectories
# ============================================================
for d in ["0", "constant", "system"]:
    (OUTPUT / d).mkdir(exist_ok=True)


def write_file(rel_path, content):
    """写入OpenFOAM文件，自动添加FoamFile头部"""
    path = OUTPUT / rel_path
    
    # 提取location和object
    parts = rel_path.split("/")
    obj = parts[-1]
    loc = "/".join(parts[:-1]) if len(parts) > 1 else ""
    if not loc:
        loc = parts[0] if parts[0] in ["0","constant","system"] else ""
    
    header = dedent(f"""\
    /*--------------------------------*- C++ -*----------------------------------*\\
      =========                 |
      \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
       \\\\    /   O peration     | 
        \\\\  /    A nd           | DEC Pelton Turbine Simulation
         \\\\/     M anipulation  | Based on Xiao & Liang (2021, 2024)
    -------------------------------------------------------------------------------
    License: DEC PELTON 2024 Research Use Only
    \\*---------------------------------------------------------------------------*/
    FoamFile
    {{
        version     2.0;
        format      ascii;
        class       {_get_class(obj, loc)};
        location    "{loc}";
        object      {obj};
    }}
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    """)
    
    with open(path, 'w') as f:
        f.write(header + content)


def _get_class(obj, loc):
    """根据文件名和位置推断OpenFOAM class"""
    class_map = {
        "U": "volVectorField",
        "p_rgh": "volScalarField",
        "alpha.water": "volScalarField",
        "k": "volScalarField",
        "omega": "volScalarField",
        "nut": "volScalarField",
        "epsilon": "volScalarField",
        "controlDict": "dictionary",
        "fvSchemes": "dictionary",
        "fvSolution": "dictionary",
        "transportProperties": "dictionary",
        "turbulenceProperties": "dictionary",
        "g": "uniformDimensionedVectorField",
    }
    return class_map.get(obj, "dictionary")


# ============================================================
# 1. system/controlDict
# ============================================================
write_file("system/controlDict", dedent("""\
application     interFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.01;

deltaT          7.89e-5;

writeControl    adjustableRunTime;
writeInterval   0.001;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;

adjustTimeStep  yes;
maxCo           0.5;
maxAlphaCo      0.3;
maxDeltaT       2e-4;
"""))

# ============================================================
# 2. system/fvSchemes
# ============================================================
write_file("system/fvSchemes", dedent("""\
ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    div(rhoPhi,U)   Gauss limitedLinearV 1;
    div(phi,alpha)  Gauss vanLeer;
    div(phi,k)      Gauss limitedLinear 1;
    div(phi,omega)  Gauss limitedLinear 1;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
"""))

# ============================================================
# 3. system/fvSolution
# ============================================================
write_file("system/fvSolution", dedent("""\
solvers
{
    "alpha.water.*"
    {
        nAlphaCorr      2;
        nAlphaSubCycles 1;
        cAlpha          1;
        MULESCorr       yes;
        nLimiterIter    5;
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0;
    }

    p_rgh
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-7;
        relTol          0.01;
    }

    p_rghFinal
    {
        $p_rgh;
        relTol          0;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0;
    }

    k
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-7;
        relTol          0.1;
    }

    omega
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-7;
        relTol          0.1;
    }
}

PIMPLE
{
    momentumPredictor no;
    nOuterCorrectors 1;
    nCorrectors     3;
    nNonOrthogonalCorrectors 0;
}
"""))

# ============================================================
# 4. constant/transportProperties
# ============================================================
write_file("constant/transportProperties", dedent("""\
phases (water air);

water
{
    transportModel  Newtonian;
    nu              1.004e-06;
    rho             998.2;
}

air
{
    transportModel  Newtonian;
    nu              1.48e-05;
    rho             1.225;
}

sigma            0.072;
"""))

# ============================================================
# 5. constant/turbulenceProperties
# ============================================================
write_file("constant/turbulenceProperties", dedent("""\
simulationType  RAS;

RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
"""))

# ============================================================
# 6. constant/g
# ============================================================
write_file("constant/g", dedent("""\
dimensions      [0 1 -2 0 0 0 0];
value           (0 0 -9.81);
"""))

# ============================================================
# 7. Initial conditions - 0/alpha.water
# ============================================================
write_file("0/alpha.water", dedent("""\
dimensions      [0 0 0 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform 1;
    }
    outlet
    {
        type            inletOutlet;
        inletValue      uniform 0;
        value           uniform 0;
    }
    walls
    {
        type            zeroGradient;
    }
    atmosphere
    {
        type            inletOutlet;
        inletValue      uniform 0;
        value           uniform 0;
    }
}
"""))

# ============================================================
# 8. Initial conditions - 0/p_rgh
# ============================================================
write_file("0/p_rgh", dedent("""\
dimensions      [1 -1 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            fixedFluxPressure;
        value           uniform 0;
    }
    outlet
    {
        type            prghPressure;
        p               uniform 0;
        value           uniform 0;
    }
    walls
    {
        type            fixedFluxPressure;
        value           uniform 0;
    }
    atmosphere
    {
        type            prghPressure;
        p               uniform 0;
        value           uniform 0;
    }
}
"""))

# ============================================================
# 9. Initial conditions - 0/U
# ============================================================
write_file("0/U", dedent("""\
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (0 0 -106.7);  // V_jet = sqrt(2gH)
    }
    outlet
    {
        type            zeroGradient;
    }
    walls
    {
        type            noSlip;
    }
    atmosphere
    {
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }
}
"""))

# ============================================================
# 10. Initial conditions - turbulence (0/k, 0/omega, 0/nut)
# ============================================================
I_turb = 0.05  # 5% turbulence intensity
U_ref = 106.7
k_val = 1.5 * (I_turb * U_ref) ** 2  # ~42.7
omega_val = k_val ** 0.5 / (0.1 * 0.02)  # length scale ~0.02m
nut_val = 0.0

write_file("0/k", f"""\
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k_val:.2f};
boundaryField
{{
    inlet       {{ type fixedValue; value uniform {k_val:.2f}; }}
    outlet      {{ type zeroGradient; }}
    walls       {{ type kqRWallFunction; value uniform {k_val:.2f}; }}
    atmosphere  {{ type inletOutlet; inletValue uniform {k_val:.2f}; value uniform {k_val:.2f}; }}
}}
""")

write_file("0/omega", f"""\
dimensions      [0 0 -1 0 0 0 0];
internalField   uniform {omega_val:.2f};
boundaryField
{{
    inlet       {{ type fixedValue; value uniform {omega_val:.2f}; }}
    outlet      {{ type zeroGradient; }}
    walls       {{ type omegaWallFunction; value uniform {omega_val:.2f}; }}
    atmosphere  {{ type inletOutlet; inletValue uniform {omega_val:.2f}; value uniform {omega_val:.2f}; }}
}}
""")

write_file("0/nut", f"""\
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
    inlet       {{ type calculated; value uniform 0; }}
    outlet      {{ type calculated; value uniform 0; }}
    walls       {{ type nutkWallFunction; value uniform 0; }}
    atmosphere  {{ type calculated; value uniform 0; }}
}}
""")

print(f"""
✅ OpenFOAM Case 生成完成!

   📁 {OUTPUT}
   
   📋 下一步:
   1. 复制到WSL: cp -r /mnt/b/undertaking/学业/非专业相关/比赛/挑战杯/水轮机/workspace/of_case/pelton_jet ~/
   2. 生成网格: cd ~/pelton_jet && blockMesh
   3. 运行仿真: interFoam
""")

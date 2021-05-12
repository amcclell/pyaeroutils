# This serves more as a template, as opposed to truly meaning to be general, as it will probably be faster to just make the modifications
# needed on a per simulation basis

def writeFluidFile(name: str, id: int, alpha: float, h: float, v: float, sol: str, restart: str):
  # Writes an AERO-F 2 input file for steady analysis
  # Inputs: name = filename (string)
  #         id = postfix for different input files, e.g. FluidFile0, and files read by the input file, e.g. Solution.0.SOL (int)
  #         alpha = angle of attack (float)
  #         h = altitude in meters (float)
  #         v = freestream velocity (float)
  #         sol = solution vector location
  #         restart = restart data location
  fn = '{}{:d}'.format(name, id)
  hFt = h / 0.3048
  content = '''// ----------------------------------------------------------------------
// PROJECT NAME:       FlightTest
// SIMULATION NAME:    InviscidSLF{id:d}
// FLUID GEOMETRY:     sources/MeshFluid.top
// ----------------------------------------------------------------------

under Problem {{
  Type   = Steady;
  Mode   = Dimensional;
  Framework = BodyFitted;
}}

under Input {{
  Prefix = "";
  GeometryPrefix = "/oak/stanford/groups/cfarhat/amcclell/FRG/MPC_SI/data/MeshFluid";
  Solution = "{sol:s}";
  RestartData = "{restart:s}";
}}

under Output {{
  under Postpro {{
    Prefix   = "results/";
    XM = 2.073180258000000;
    YM = 0.0;
    ZM =  1.327244488000000;
    Pressure = "Pressure.{id:d}.bin";
    Density = "Density.{id:d}.bin";
    Velocity = "Velocity.{id:d}.bin";
    Force = "Force.{id:d}.txt";
    LiftandDrag = "LiftandDrag.{id:d}.txt";
    Width = 24;
    Precision = 16;
  }}
  under Restart {{
    Frequency = 0;
    Prefix   = "references/";
    FilePackage = "";
    Solution = "Solution.{id:d}.SOL";
    RestartData = "Restart.{id:d}.RST";
  }}
}}

under ReferenceState {{
  Velocity = 30.48;
  Altitude = 1000.0;
  AltitudeUnit = FT;
  ProblemUnit = M;
}}

under Equations {{
  Type = Euler;
  under FluidModel[0] {{
    Fluid = PerfectGas;
    under GasModel {{
      SpecificHeatRatio = 1.4; 
    }}
  }}
}}

under BoundaryConditions {{
  under Inlet {{
    Type    = External;
    Velocity = {v:.16e};
    Alpha = {AoA:.16e};
    Beta    = 0.0;
    Altitude = {alt:.16e};
    AltitudeUnit = FT;
    ProblemUnit = M;
  }}
}}

under Space {{
  under NavierStokes {{
    Flux              = Roe;
    Reconstruction    = Linear;
    AdvectiveOperator = FiniteVolume;
    Limiter           = VanAlbada;
    Gradient          = LeastSquares;
    Beta              = 0.333333333333;
    Gamma             = 1.0;
  }}
  under Boundaries {{
    Type = StegerWarming;
  }}
}}

under Time {{
  Type = Implicit;
  MaxIts = 10000;
  MinimumRestartIterations = 1;
  Eps = 1e-8;
  EpsAbs = 1e-4;
  CheckLinearSolver = Off;
  under CflLaw {{
    Strategy = Hybrid;
    Cfl0 = 5.0;
    Cfl1 = 1.0;
    CflMax = 50000;
  }}
  under Implicit {{
    Type = BackwardEuler;
    MatrixVectorProduct = FiniteDifference;
    FiniteDifferenceOrder = FirstOrder;
    under Newton {{
      MaxIts = 4;
      Eps = 1e-3;
      EpsAbsRes = 1e-4;
      EpsAbsInc = 1e-6;
      FailSafe = On;
      under LinearSolver {{
        under NavierStokes {{
          Type = Gmres;
          //Output = "stdout";
          MaxIts = 300;
          KrylovVectors = 300;
          Eps = 1e-4;
          AbsoluteEps = 1e-6;
          KspPrecJacobian = H2;
          under Preconditioner {{
            Type = Ras;
            Fill = 0;
          }}
        }}
      }}
    }}
  }}
}}
'''.format(id=id, AoA=alpha, alt=hFt, v=v, sol=sol, restart=restart)
  with open(fn, 'w') as f:
    print('File {} is open...'.format(fn))
    f.write(content)
  print('File {} is closed.'.format(fn))

def writeFluidFileSA(name: str, id: int, alpha: float, h: float, v: float, sol: str):
  # Writes an AERO-F 2 input file for steady sensitivity analysis
  # Inputs: name = filename (string)
  #         id = postfix for different input files, e.g. FluidFile0, and files read by the input file, e.g. Solution.0.SOL (int)
  #         alpha = angle of attack (float)
  #         h = altitude in meters (float)
  #         v = freestream velocity (float)
  #         sol = solution vector location
  fn = '{}{}{:d}'.format(name, 'SA', id)
  hFt = h / 0.3048
  content = '''// ----------------------------------------------------------------------
// PROJECT NAME:       FlightTest
// SIMULATION NAME:    InviscidSLFSA{id:d}
// FLUID GEOMETRY:     sources/MeshFluid.top
// ----------------------------------------------------------------------

under Problem {{
  Type   = SensitivityAnalysis;
  Mode   = Dimensional;
  Framework = BodyFitted;
}}

under Input {{
  Prefix = "";
  GeometryPrefix = "/oak/stanford/groups/cfarhat/amcclell/FRG/MPC_SI/data/MeshFluid";
  Solution = "{sol:s}";
  ShapeDerivative = "/oak/stanford/groups/cfarhat/amcclell/FRG/MPC_SI/simulations/Hack1DegCSModes/references/MeshFluid.1deg.CSModes";
}}

under Output {{
  under Postpro {{
    Prefix   = "results/";
    XM = 2.073180258000000;
    YM = 0.0;
    ZM =  1.327244488000000;
    ForceSensitivity = "ForceSensitivity.{id:d}.txt";
    LiftandDragSensitivity = "LiftandDragSensitivity.{id:d}.txt";
    Width = 24;
    Precision = 16;
  }}
  under Restart {{
    Frequency = 0;
    Prefix   = "references/";
    FilePackage = "";
    Solution = "Solution.{id:d}.SOL";
    RestartData = "Restart.{id:d}.RST";
  }}
}}

under ReferenceState {{
  Velocity = 30.48;
  Altitude = 1000.0;
  AltitudeUnit = FT;
  ProblemUnit = M;
}}

under Equations {{
  Type = Euler;
  under FluidModel[0] {{
    Fluid = PerfectGas;
    under GasModel {{
      SpecificHeatRatio = 1.4; 
    }}
  }}
}}

under BoundaryConditions {{
  under Inlet {{
    Type    = External;
    Velocity = {v:.16e};
    Alpha = {AoA:.16e};
    Beta    = 0.0;
    Altitude = {alt:.16e};
    AltitudeUnit = FT;
    ProblemUnit = M;
  }}
  under Wall {{
    Type = Adiabatic;
    Integration = WallFunction;
    Delta = 0.00116;
  }}
}}

under Space {{
  under NavierStokes {{
    Flux              = Roe;
    Reconstruction    = Linear;
    AdvectiveOperator = FiniteVolume;
    Limiter           = VanAlbada;
    Gradient          = LeastSquares;
    Beta              = 0.333333333333;
    Gamma             = 1.0;
  }}
  under Boundaries {{
    Type = StegerWarming;
  }}
}}

under Time {{
  Type = Implicit;
  MaxIts = 1000;
  Eps = 1e-6;
  EpsAbs = 1e-2;
  CheckLinearSolver = Off;
  under CflLaw {{
    Strategy = Hybrid;
    Cfl0 = 100.0;
    Cfl1 = 1.0;
    CflMax = 10000;
  }}
  under Implicit {{
    Type = BackwardEuler;
    MatrixVectorProduct = FiniteDifference;
    //ExactImplementation = New;
    under Newton {{
      MaxIts = 1;
      Eps = 1e-2;
      EpsAbsRes = 1e-4;
      EpsAbsInc = 1e-6;
      FailSafe = On;
      under LinearSolver {{
        under NavierStokes {{
          Type = Gmres;
          //Output = "stdout";
          MaxIts = 150;
          KrylovVectors = 150;
          Eps = 1e-4;
          AbsoluteEps = 1e-6;
          KspPrecJacobian = H1;
          under Preconditioner {{
            Type = Ras;
            Fill = 1;
          }}
        }}
      }}
    }}
  }}
}}

under SensitivityAnalysis {{
  Method = Direct;
  SparseApproach = On;
  MatrixVectorProduct = Exact;
  SensitivityComputation = Analytical;
  SensitivityMesh = On;
  SensitivityAlpha = On;
  //Debug = On;
  //SensitivityOutput = "results/SensitivityOutput.{id:d}.txt";
  MaxOuterIts = 20;
  NumShapeVariables = 12;
  ModalInput = On;
  under LinearSolver {{
    Type = Gmres;
    MaxIts = 500;
    KrylovVectors = 500;
    Eps = 1e-8;
    AbsoluteEps = 1e-4;
    KspPrecJacobian = H2;
    under Preconditioner {{
      Type = Ras;
      Fill = 0;
    }}
  }}
}}
'''.format(id=id, AoA=alpha, alt=hFt, v=v, sol=sol)
  with open(fn, 'w') as f:
    print('File {} is open...'.format(fn))
    f.write(content)
  print('File {} is closed.'.format(fn))

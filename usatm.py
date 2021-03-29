import numpy as np
from enum import Enum

class Unit(Enum):
  '''Unit enumerated type'''
  CM = 0
  M = 1
  KM = 2
  IN = 3
  FT = 4
  MI = 5

class USStandardAtmosphere(object):
  '''US Standard Atmosphere'''
  r0 = 6356766.0 # effective Earth radius at 45 deg North latitude (m)
  g0 = 9.80665   # sea-level acceleration due to gravity at 45 deg North (m/s^2)
  M0 = 28.9644   # molecular weight of air (kg/kmol)
  T0 = 288.15    # sea-level temperature (K)
  Ru = 8.31432e3 # Universal gas constant (J/kmol*K)
  R = Ru / M0      # Gas constant of air (J/kg*K)
  P0 = 101325.0  # sea-level pressure (N/m^2)

  numSegments = 7 # number of segments in piecewise linear temperature function
  Z_max = 86.0e3 # maximum altitude accounted for (actual standard goes to 1000 km) (m)
  Hb_max = r0 * Z_max / (r0 + Z_max) # maximum geopotential altitude (m)
  L = np.array([-6.5e-3, 0, 1e-3, 2.8e-3, 0, -2.8e-3, -2.0e-3])  # slope of piecewise linear temperature function segments (K/m)
  Hb = np.array([0.0, 11.0e3, 20.0e3, 32.0e3, 47.0e3, 51.0e3, 71.0e3]) # geopotential heights at segment start (m)
  Pb = np.empty((numSegments, ), dtype=np.float)   # Pressures at segment start (N/m^2)
  Tmb = np.empty((numSegments, ), dtype=np.float)  # Temperatures at segment start (K)

  g0dR = g0 / R
  g0dRL = np.empty((numSegments, ), dtype=np.float) # stores g0/(R*L[i])

  numMdM0Segments = 12 # number of data points in the molecular weight ratio table
  MdM0 = np.array([1.0, 0.999996, 0.999989, 0.999971, 0.999941, 0.999909, 0.999870,
                        0.999829, 0.999786, 0.999741, 0.999694, 0.999641, 0.999579]) # ratio of molecular weight to sea-level value altitudes between 80 km and 86 km
  Z = np.array([80000.0, 80500.0, 81000.0, 81500.0, 82000.0, 82500.0, 83000.0,
                          83500.0, 84000.0, 84500.0, 85000.0, 85500.0, 86000.0]) # Corresponding altitudes for the above ratios (m)
  slopeMdM0 = np.empty((numMdM0Segments, ), dtype=np.float) # slopes used for linear interpolation of MdM0 (m^-1)

  # These quantities may not be needed for the boundary conditions, but could be used to overwrite other parameters
  # of the simulation to ensure consistent models
  gamma = 1.4 # Specific heat ratio of air; if the US Stan. Atm. is used, will assume air everywhere
              # (need to set parser to force the gas model to be this way)
  beta = 1.458e-6 # Sutherland's constant for air (AERO-F manual calls this mu_0; units kg/m*s*K^(1/2))
  S = 110.4 # Sutherland reference temperature for air (K)

  # Specific heat at constant pressure
  Cp = gamma * R / (gamma - 1)

  def __init__(self, alt_unit: Unit = Unit.M, prob_unit: Unit = Unit.M):

    # Conversion factors for converting given altitude units to meters, and from meters to problem conversion units
    self.altitude_unit = alt_unit
    self.problem_unit = prob_unit
    self.altitude_conversion = 1.
    self.problem_rho_conversion = 1.
    self.problem_P_conversion = 1.
    self.problem_T_conversion = 1.
    self.problem_Length_conversion = 1.
    self.problem_mu0_conversion = 1.
    self.problem_R_conversion = 1.

    self.__initializeUSStandardAtmosphere()
    self.__setConversions()

  # Initialize all arrays above
  def __initializeUSStandardAtmosphere(self):
    self.Tmb[0] = self.T0
    self.Pb[0] = self.P0
    self.g0dRL[0] = self.g0dR / self.L[0]

    for j in np.arange(1, self.numSegments):
      self.Tmb[j] = self.Tmb[j - 1] + self.L[j - 1] * (self.Hb[j] - self.Hb[j - 1])
      if self.L[j - 1] != 0:
        self.Pb[j] = self.Pb[j - 1] * np.power(self.Tmb[j - 1] / self.Tmb[j], self.g0 / (self.R * self.L[j - 1]))
      else:
        self.Pb[j] = self.Pb[j - 1] * np.exp(-self.g0 * (self.Hb[j] - self.Hb[j - 1]) / (self.R * self.Tmb[j - 1]))
      if self.L[j] != 0:
        self.g0dRL[j] = self.g0dR / self.L[j]
      else:
        self.g0dRL[j] = 0.0

    # Initialize slopes used for linear interpolation of the ratio of the molecular weight to that at sea-level
    for j in range(self.numMdM0Segments):
      self.slopeMdM0[j] = (self.MdM0[j + 1] - self.MdM0[j]) / (self.Z[j + 1] - self.Z[j])

  # Reset conversion factors to desired units
  def __setConversions(self):
    if self.altitude_unit is Unit.CM:
      self.altitude_conversion = 1.0e-2
    elif self.altitude_unit is Unit.M:
      self.altitude_conversion = 1.0
    elif self.altitude_unit is Unit.KM:
      self.altitude_conversion = 1.0e3
    elif self.altitude_unit is Unit.IN:
      self.altitude_conversion = 2.54e-2
    elif self.altitude_unit is Unit.FT:
      self.altitude_conversion = 3.048e-1
    elif self.altitude_unit is Unit.MI:
      self.altitude_conversion = 1.609344e3
    else:
      raise ValueError('*** Error: Invalid altitude unit')

    # Convert US Standard Atmosphere output into system of units desired supported units are meters, inches and feet
    # conversions are from SI to desired units (so if meters is chosen, no conversions are necessary)
    if self.altitude_unit is Unit.M:
      # Final Units:
      # Density: kg/m^3; Pressure: N/m^2; Temperature: Kelvin; Length: m; Sutherland's constant mu0: kg/m*s*K^(1/2);
      # Ideal gas constant: J/kg*Kelvin = m^2/s^2*K
      self.problem_rho_conversion = 1.0
      self.problem_P_conversion = 1.0
      self.problem_T_conversion = 1.0
      self.problem_Length_conversion = 1.0
      self.problem_mu0_conversion = 1.0
      self.problem_R_conversion = 1.0
    elif self.altitude_unit is Unit.IN:
      # Final Units: Note that 1 slinch = 12 slugs = 12 * 14.59390 kg
      # Density: slinch/in^3; Pressure: lbf/in^2 = slinch/in*s^2; Temperature: Rankine; Length: in; Sutherland's
      # constant mu0: slinch/in*s*R^(1/2); Ideal gas constant: in^2/s^2*R
      self.problem_rho_conversion = np.power(2.54e-2, 3) / (14.59390 * 12)
      self.problem_P_conversion = 2.54e-2 / (14.59390 * 12)
      self.problem_T_conversion = 1.8
      self.problem_Length_conversion = 1 / 2.54e-2
      self.problem_mu0_conversion = 2.54e-2 / (14.59390 * 12 * np.sqrt(1.8))
      self.problem_R_conversion = 1 / (np.power(2.54e-2, 2) * 1.8)
    elif self.altitude_unit is Unit.FT:
      # Final Units:
      # Density: slug/ft^3; Pressure: lbf/ft^2 = slug/ft*s^2; Temperature: Rankine; Length: ft; Sutherland's constant
      # mu0: slug/ft*s*R^(1/2); Ideal gas constant: ft^2/s^2*R
      self.problem_rho_conversion = np.power(3.048e-1, 3)/(14.59390)
      self.problem_P_conversion = 3.048e-1 / (14.59390)
      self.problem_T_conversion = 1.8
      self.problem_Length_conversion = 1 / 3.048e-1
      self.problem_mu0_conversion = 3.048e-1 / (14.59390 * np.sqrt(1.8))
      self.problem_R_conversion = 1 / (np.power(3.048e-1, 2) * 1.8)
    else:
      raise ValueError("*** Error: Only ProblemUnit = {M, IN, FT} are currently accepted. Exiting...\n")

  def resetConversions(self, alt_unit: Unit = Unit.M, prob_unit: Unit = Unit.M):
    self.altitude_unit = alt_unit
    self.problem_unit = prob_unit
    self.__setConversions()
  
  # Given altitude, calculate density, pressure, temperature and speed of sound (all dimensional quantities);
  # altitude should be in self.altitude_unit
  def calculateUSStandardAtmosphere(self, z: float):
    z_SI = z * self.altitude_conversion
    if z_SI > self.Z_max:
      raise ValueError('*** Error: {} m greater than max altitude {} m'.format(z_SI, self.Z_max))
    h = self.r0 * z_SI / (self.r0 + z_SI)
    if self.Hb[0] <= h and h <= self.Hb[1]:
      T = self.Tmb[0] + self.L[0] * h
      P = self.Pb[0] * np.power(self.Tmb[0] / T, self.g0dRL[0])
      rho = P / (self.R * T)
    elif self.Hb[1] < h and h <= self.Hb[2]:
      T = self.Tmb[1]
      P = self.Pb[1] * np.exp(-self.g0dR * (h - self.Hb[1]) / T)
      rho = P / (self.R * T)
    elif self.Hb[2] < h and h <= self.Hb[3]:
      T = self.Tmb[2] + self.L[2] * h
      P = self.Pb[2] * np.power(self.Tmb[2] / T, self.g0dRL[2])
      rho = P / (self.R * T)
    elif self.Hb[3] < h and h <= self.Hb[4]:
      T = self.Tmb[3] + self.L[3] * h
      P = self.Pb[3] * np.power(self.Tmb[3] / T, self.g0dRL[3])
      rho = P / (self.R * T)
    elif self.Hb[4] < h and h <= self.Hb[5]:
      T = self.Tmb[4]
      P = self.Pb[4] * np.exp(-self.g0dR * (h - self.Hb[4]) / T)
      rho = P / (self.R * T)
    elif self.Hb[5] < h and h <= self.Hb[6]:
      T = self.Tmb[5] + self.L[5] * h
      P = self.Pb[5] * np.power(self.Tmb[5] / T, self.g0dRL[5])
      rho = P / (self.R * T)
    else:
      T = self.Tmb[6] + self.L[6] * h
      P = self.Pb[6] * np.power(self.Tmb[6] / T, self.g0dRL[6])
      rho = P / (self.R * T)

      mdm0 = 1.0
      for j in range(self.numMdM0Segments):
        if self.Z[j] < z_SI and z_SI <= self.Z[j+1]:
          mdm0 = self.slopeMdM0[j] * (z_SI - self.Z[j]) + self.MdM0[j]
          break
      T = mdm0 * T
    
    c = np.sqrt(self.gamma * P / rho)
    g = self.g0 * (self.r0 / (self.r0 + z_SI)) ** 2
    mu = self.beta * np.sqrt(T) / (1. + self.S / T)
    kappa = 2.64638e-3 * T ** (1.5) / (T + 245.4 * 10 ** (-12 / T ))

    return rho, P, T, c, g, mu, kappa

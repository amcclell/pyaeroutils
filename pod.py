import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer

from importlib.util import find_spec
matlabSpec = find_spec('matlab')
if matlabSpec is not None:
  foundMatlab = True
else:
  foundMatlab = False

if foundMatlab:
  import matlab
  import matlab.engine

def readSV(file: str):
  sv = np.loadtxt(file, skiprows=2)
  eng = sv * sv
  reng = np.cumsum(eng / np.sum(eng))

  return sv, eng, reng

def analyzeBasis(file: str, tol: float, verbose: bool = False, plot: bool = False):
  sv, eng, reng = readSV(file)
  ind = np.arange(sv.size)

  isTol = (1 - reng <= tol)
  if ind[isTol].size != 0:
    nV = ind[isTol][0] + 1
  else:
    nV = ind[-1]

  if verbose:
    print('File: {}'.format(file))
    print('Vectors required to meet tolerance of {:e}: {}'.format(tol, nV))

  if plot:
    fig1, ax1 = plt.subplots()

    ax1.semilogy(ind, sv)
    ax1.set_xlabel(r'Index $(i)$')
    ax1.set_ylabel(r'$\sigma_i$')
    ax1.set_title('File: {}, Singular Values'.format(os.path.basename(file)))

    fig2, ax2 = plt.subplots()

    ax2.semilogy(ind, 1 - reng)
    ax2.set_xlabel(r'Index $(i)$')
    ax2.set_ylabel(r'$1 - \varepsilon_{rel}$')
    ax2.set_title('File: {}, Truncated Relative Energy'.format(os.path.basename(file)))

    #plt.show()

    return sv, eng, reng, nV, fig1, ax1, fig2, ax2
  
  return sv, eng, reng, nV

def readROMAero(file: str):
  with open(file) as f:
    ROMSize = np.genfromtxt([f.readline().strip()], dtype=np.int)
  nF = ROMSize[0]
  nS = ROMSize[1]
  romAero = np.loadtxt(file, skiprows=1, dtype=np.float)
  return romAero, nF, nS

def readROMAerodFdX(file: str):
  romAero, nF, nS = readROMAero(file)
  filedFdX = file + '.dFdX'
  try:
    dFdX = np.loadtxt(filedFdX, skiprows=1, dtype=np.float)
  except IOError:
    print('{} not found.'.format(filedFdX))
    dFdX = None
  if dFdX is not None:
    if dFdX.shape[0] != nS or dFdX.shape[1] != nS:
      raise ValueError('Shape mismatch for dFdX: dFdX.shape = {}, nS = {}'.format(dFdX.shape, nS))
  return romAero, dFdX, nF, nS

def readROMAeroCtrlSurf(file: str):
  romAero, dFdX, nF, nS = readROMAerodFdX(file)
  fileCtrlSurf = file + '.CtrlSurf'
  try:
    ctrlSurfBlock = np.loadtxt(fileCtrlSurf, skiprows=1, dtype=np.float)
  except IOError:
    print('{} not found.'.format(fileCtrlSurf))
    ctrlSurfBlock = None
  if ctrlSurfBlock is not None:
    if ctrlSurfBlock.shape[0] != romAero.shape[0]:
      raise ValueError('Shape mismatch for ctrlSurfBlock: ctrlSurfBlock.shape[0] = {}, romAero.shape[0] = {}'.format(ctrlSurfBlock.shape[0], romAero.shape[0]))
  return romAero, dFdX, ctrlSurfBlock, nF, nS

def checkStability(mat):
  ev = np.linalg.eigvals(mat)
  if np.any(np.real(ev) >= 0.):
    return False
  else:
    return True

def checkSubROMStability(romAero, nF, gamma, rhoref, Pref, Vref):
  cref = np.sqrt(gamma * Pref / rhoref)
  Mref = Vref / cref
  cnst = Mref * np.sqrt(gamma)
  H = romAero[0:nF, 0:nF] / cnst

  isStable = np.empty((nF, ), dtype=np.bool)
  for i in range(nF):
    isStable[i] = checkStability(H[0:(i + 1), 0:(i + 1)])
  return isStable

def findClosestStable(isStable, nV, nT):
  ind = np.arange(nT)[isStable] + 1
  indM = ind - nV
  tmp = indM <= 0
  if np.any(tmp):
    indL = nV - np.abs(indM[tmp]).min()
  else:
    indL = None
  tmp = indM >= 0
  if np.any(tmp):
    indU = nV + indM[tmp].min()
  else:
    indU = None
  return indL, indU

def chooseStableSize(indL, indU, nV):
  if indL is not None:
    dL = nV - indL
  else:
    dL = None
  if indU is not None:
    dU = indU - nV
  else:
    dU = None
  if dL is not None and dU is not None:
    return [indL, indU][np.argmin([dL, dU])]
  elif dL is not None:
    return indL
  elif dU is not None:
    return indU
  else:
    print('*** Warning: no stable ROM size given. Using desired size.')
    return nV

def extractSubROM(romAero, nF, nS, nFNew):
  ind = np.concatenate((np.arange(nFNew), np.arange(nF, nF + 2 * nS)))
  ind2d = np.ix_(ind, ind)
  romAeroNew = romAero[ind2d]
  return romAeroNew

def extractSubROMdFdX(romAero, dFdX, nF, nS, nFNew):
  return extractSubROM(romAero, nF, nS, nFNew), dFdX

def extractSubROMCtrlSurf(romAero, dFdX, ctrlSurfBlock, nF, nS, nFNew):
  romAeroNew, dFdXNew = extractSubROMdFdX(romAero, dFdX, nF, nS, nFNew)
  ind = np.concatenate((np.arange(nFNew), np.arange(nF, nF + 2 * nS)))
  ctrlSurfBlockNew = ctrlSurfBlock[ind, :]
  return romAeroNew, dFdXNew, ctrlSurfBlockNew

def extractOperators(romAero, nF, nS):
  '''return H, B, C, P, K
  Assumes that no mass matrix inverse has been applied.'''
  H = -romAero[0:nF, 0:nF]
  B = -romAero[0:nF, nF:(nF + nS)]
  C = -romAero[0:nF, (nF + nS):(nF + 2 * nS)]
  P = romAero[nF:(nF + nS), 0:nF]
  K = -romAero[nF:(nF + nS), (nF + nS):(nF + 2 * nS)]
  return H, B, C, P, K

def extractOperatorsdFdX(romAero, dFdX, nF, nS):
  '''return H, B, C, P, K, Py
  Assumes that no mass matrix inverse has been applied.'''
  H, B, C, P, K = extractOperators(romAero, nF, nS)
  Py = dFdX.copy()
  return H, B, C, P, K, Py

def extractOperatorsCtrlSurf(romAero, dFdX, ctrlSurfBlock, nF, nS):
  '''return H, B, C, P, K, Py, Bcs, Ccs, Pycs
  Assumes that no mass matrix inverse has been applied.'''
  H, B, C, P, K, Py = extractOperatorsdFdX(romAero, dFdX, nF, nS)
  nCS = ctrlSurfBlock.shape[1] // 2
  Bcs = -ctrlSurfBlock[0:nF, 0:nCS]
  Ccs = -ctrlSurfBlock[0:nF, nCS:]
  Pycs = ctrlSurfBlock[nF:(nF + nS), nCS:]

def writeROMAero(file, romAero, nF, nS):
  np.savetxt(file, romAero, fmt='% .16e', header='{} {}'.format(nF, nS), comments='')

def writeROMAerodFdX(file, romAero, nF, nS, dFdX = None):
  writeROMAero(file, romAero, nF, nS)
  if dFdX is not None:
    filedFdX = file + '.dFdX'
    np.savetxt(filedFdX, dFdX, fmt='% .16e', header='{} {}'.format(nS, nS), comments='')

def writeROMAeroCtrlSurf(file, romAero, nF, nS, dFdX = None, ctrlSurfBlock = None):
  writeROMAerodFdX(file, romAero, nF, nS, dFdX)
  if ctrlSurfBlock is not None:
    fileCtrlSurf = file + '.CtrlSurf'
    np.savetxt(fileCtrlSurf, ctrlSurfBlock, fmt='% .16e', header='{} {}'.format(ctrlSurfBlock.shape[0], ctrlSurfBlock.shape[1]), comments='')

def dimROMAero(rom, nF, nS, p_inf, rho_inf):
  romDim = rom.copy()
  romDim[0:nF, 0:nF] *= np.sqrt(p_inf / rho_inf)
  romDim[0:nF, (nF + nS):(nF + 2 * nS)] *= np.sqrt(p_inf / rho_inf)
  romDim[nF:(nF + nS), 0:nF] *= p_inf
  return romDim

def dimROMAerodFdX(rom, nF, nS, p_inf, rho_inf, dFdX = None):
  romDim = dimROMAero(rom, nF, nS, p_inf, rho_inf)
  if dFdX is not None:
    dFdXDim = dFdX * p_inf
  else:
    dFdXDim = None
  return romDim, dFdXDim

def dimROMAeroCtrlSurf(rom, nF, nS, p_inf, rho_inf, dFdX = None, ctrlSurfBlock = None):
  romDim, dFdXDim = dimROMAerodFdX(rom, nF, nS, p_inf, rho_inf, dFdX)
  if ctrlSurfBlock is not None:
    nCtrlSurf = ctrlSurfBlock.shape[1] // 2
    ctrlSurfBlockDim = ctrlSurfBlock.copy()
    ctrlSurfBlockDim[0:nF, nCtrlSurf:] *= np.sqrt(p_inf / rho_inf)
    ctrlSurfBlockDim[nF:(nF + nS), nCtrlSurf:] *= p_inf
  else:
    ctrlSurfBlockDim = None
  return romDim, dFdXDim, ctrlSurfBlockDim

def addMassAndStiffness(rom, mass, stiffness, nF, nS, dFdX = None, dFdCtrl = None, ctrlSurfBlock = None):
  A = rom.copy()
  A[nF:(nF + nS), (nF + nS):(nF + 2 * nS)] = -stiffness
  if dFdX is not None:
    A[nF:(nF + nS), (nF + nS):(nF + 2 * nS)] += dFdX
  A[nF:(nF + nS), :] = np.linalg.solve(mass, A[nF:(nF + nS), :])

  B = np.empty((nF + 2 * nS, 0), dtype=np.float)
  if dFdCtrl is not None:
    nC = dFdCtrl.shape[1]
    B = np.zeros((nF + 2 * nS, nC), dtype=np.float)
    B[nF:(nF + nS), :] = dFdCtrl

  if ctrlSurfBlock is not None:
    B = np.hstack((B, ctrlSurfBlock))
  
  if B.shape[1] != 0:
    B[nF:(nF + nS), :] = np.linalg.solve(mass, B[nF:(nF + nS), :])
  else:
    B = None
  
  return A, B

def checkMatrix(mat):
  if len(mat.shape) < 2:
    mat = np.atleast_2d(mat).T
  return mat

def readMatrixData(filename: str):
  mass = None
  stiffness = None
  dFdCtrl = None
  with open(filename) as f:
    lines = np.array(f.readlines())
    for line in lines[1:]:
      tmp = line.strip().split()
      file = tmp[0]
      id = np.int(tmp[1])
      if id == 0:
        stiffness = np.loadtxt(file, skiprows=1)
      elif id == 1:
        mass = np.loadtxt(file, skiprows=1)
      elif id == 2:
        dFdCtrl = np.loadtxt(file, skiprows=1)
      else:
        raise ValueError('Invalid matrix ID ({}) for file {} listed in file {}'.format(id, file, filename))
  mass = checkMatrix(mass)
  stiffness = checkMatrix(stiffness)
  dFdCtrl = checkMatrix(dFdCtrl)
  return mass, stiffness, dFdCtrl
  
def forcedCtrlSurf(A, B, x0, nCtrl, nCtrlSurf, dt, nT, amp, freq, modes):
  t = np.arange(0, nT + 1, dtype=np.float) * dt
  x = np.empty((nT + 1, x0.size))
  u = np.zeros((nT + 1, B.shape[1]))

  x[0] = x0

  omg = 2 * np.pi * freq
  fexp = lambda omg, t : np.exp(-(omg * omg) * t * t)
  f = lambda amp, omg, t : amp * (1 - fexp(omg, t)) * np.sin(omg * t)
  dfdt = lambda amp, omg, t : amp * ((2. * omg * omg) * t * fexp(omg, t) * np.sin(omg * t) + omg * (1 - fexp(omg, t)) * np.cos(omg * t))

  u[:, modes + nCtrl] = np.atleast_2d(dfdt(amp, omg, t)).T
  u[:, modes + nCtrl + nCtrlSurf] = np.atleast_2d(f(amp, omg, t)).T

  op0 = np.eye(A.shape[0]) / dt - A
  op = (1.5 / dt) * np.eye(A.shape[0]) - A

  rhs = x[0] / dt + B @ u[1]
  x[1] = np.linalg.solve(op0, rhs)

  coeffn = 2.0 / dt
  coeffnm1 = 0.5 / dt
  for i in np.arange(2, nT + 1):
    rhs = coeffn * x[i - 1] - coeffnm1 * x[i - 2] + B @ u[i]
    x[i] = np.linalg.solve(op, rhs)

  return t, x, u

def forcedLinearizedROM(H: np.ndarray, B: np.ndarray, C: np.ndarray, P: np.ndarray, dt: float, nT: int, amp: np.ndarray, freq: np.ndarray,
                        dFdX: np.ndarray = None, nCtrlSurf: int = 0, Bcs: np.ndarray = None, Ccs: np.ndarray = None, ampcs: np.ndarray = None,
                        freqcs: np.ndarray = None, dFdXcs: np.ndarray = None):
  t = np.arange(0, nT + 1, dtype=np.float) * dt
  w = np.zeros((nT + 1, H.shape[1]), dtype=np.float)
  dF = np.empty((nT + 1, B.shape[1]), dtype=np.float)
  u = np.zeros((nT + 1, B.shape[1]), dtype=np.float)
  udot = np.zeros((nT + 1, B.shape[1]), dtype=np.float)

  fexp = lambda omg, t : np.exp(-(omg * omg) * (t * t))
  f = lambda amp, omg, t : amp * (1 - fexp(omg, t)) * np.sin(omg * t)
  dfdt = lambda amp, omg, t : amp * ((2. * omg * omg) * t * fexp(omg, t) * np.sin(omg * t) + omg * (1 - fexp(omg, t)) * np.cos(omg * t))

  omg = 2 * np.pi * freq
  u[0] = f(amp, omg, t[0])
  udot[0] = dfdt(amp, omg, t[0])
  u[1] = f(amp, omg, t[1])
  udot[1] = dfdt(amp, omg, t[1])

  if nCtrlSurf > 0:
    if Bcs is None or Ccs is None or ampcs is None or freqcs is None:
      raise ValueError('Missing input for nCtrlSurf > 0 ({})'.format(nCtrlSurf))
    if dFdX is not None and dFdXcs is None:
      print("*** Warning: dFdX is included but dFdXcs is not. Consider including dFdXcs as well.", flush=True)
    ucs = np.zeros((nT + 1, Bcs.shape[1]), dtype=np.float)
    ucsdot = np.zeros((nT + 1, Bcs.shape[1]), dtype=np.float)
    omgcs = 2 * np.pi * freqcs
    ucs[0] = f(ampcs, omgcs, t[0])
    ucsdot[0] = dfdt(ampcs, omgcs, t[0])
    ucs[1] = f(ampcs, omgcs, t[1])
    ucsdot[1] = dfdt(ampcs, omgcs, t[1])
  else:
    ucs = None
    ucsdot = None

  op0 = np.eye(H.shape[0]) / dt + H
  op = (1.5 / dt) * np.eye(H.shape[0]) + H

  rhs = w[0] / dt - B @ udot[1] - C @ u[1]
  if nCtrlSurf > 0:
    rhs -= Bcs @ ucsdot[1] + Ccs @ ucs[1]
  w[1] = np.linalg.solve(op0, rhs)
  dF[1] = P @ w[1]
  if dFdX is not None:
    dF[1] += dFdX @ u[1]
  if nCtrlSurf > 0 and dFdXcs is not None:
    dF[1] += dFdXcs @ ucs[1]

  coeffn = 2.0 / dt
  coeffnm1 = 0.5 / dt
  for i in np.arange(2, nT + 1):
    u[i] = f(amp, omg, t[i])
    udot[i] = dfdt(amp, omg, t[i])
    rhs = coeffn * w[i - 1] - coeffnm1 * w[i - 2] - B @ udot[i] - C @ u[i]
    if nCtrlSurf > 0:
      ucs[i] = f(ampcs, omgcs, t[i])
      ucsdot[i] = dfdt(ampcs, omgcs, t[i])
      rhs -= Bcs @ ucsdot[i] + Ccs @ ucs[i]
    w[i] = np.linalg.solve(op, rhs)
    dF[i] = P @ w[i]
    if dFdX is not None:
      dF[i] += dFdX @ u[i]
    if nCtrlSurf > 0 and dFdXcs is not None:
      dF[i] += dFdXcs @ ucs[i]

  return t, w, dF, u, udot, ucs, ucsdot

def stabilizeROM(Ma, Me, k, p, tau, mu, eng = None, **kwargs):
  if eng is None:
    Q = np.eye(k, dtype=np.float)
    P11 = cp.Variable((k, k), symmetric=True)
    P22 = cp.Variable((p, p), symmetric=True)
    P12 = cp.Variable((k, p))
    CM = cp.bmat([[P11, P12], [P12.T, P22]])
    obj = cp.Minimize(cp.norm(cp.hstack([P12.T, P22]), 'fro') + tau * cp.norm(P11, 'fro'))

    cnstrts = [CM - mu * np.eye(k + p, dtype=np.float) >> 0,
              (Me.T @ CM @ Ma + Ma.T @ CM @ Me == -Q)]
    
    prob = cp.Problem(obj, cnstrts)
    prob.solve(**kwargs)

    X = cp.bmat([[P11], [P12.T]]).value
    return X, prob.status
  else:
    MaM = matlab.double(Ma.tolist())
    MeM = matlab.double(Ma.tolist())
    kM = float(k)
    pM = float(p)
    tauM = float(tau)
    muM = float(mu)

    XM, statusM = eng.stabilizeROM(MaM, MeM, kM, pM, tauM, muM, nargout=2)
    X = np.array(XM._data.tolist())
    X = X.reshape(XM.size, order='F')
    status = str(statusM).lower()
    return X, status

def getStabilizedROM(romAero, nF, nS, nfd, tau, margin: float = 1e-8, mu: float = 1e-8, outputAll: bool = False, start: float = None, useMatlab: bool = False, **kwargs):
  if start is None:
    start = timer()
  k = nfd
  P = nF - nfd

  if useMatlab and foundMatlab:
    eng = matlab.engine.start_matlab()
    cdir = eng.cd('/home/users/amcclell/matlab/cvx/')
    eng.cvx_setup(nargout=0)
    odir = eng.cd(cdir)
    #eng.run('/home/users/amcclell/matlab/cvx/cvx_startup.m', nargout=0)
  elif useMatlab:
    print('*** Warning: "matlab" module not found. Using cvxpy instead.', flush=True)
    eng = None
  else:
    eng = None

  for p in np.arange(1, P + 1):
    Me = np.eye(k + p, k)
    Ma = romAero[0:(k + p), 0:k] + margin * Me

    X, status = stabilizeROM(Ma, Me, k, p, tau, mu, eng, **kwargs)
    print('k + p = {}, status = {}'.format(k + p, status), flush=True)

    if status not in ["infeasible", "unbounded"]:
      break
    else:
      print('Problem infeasible. Incrementing p. Elapsed Time: {}'.format(timer() - start))

  if useMatlab:
    eng.quit()
  
  if status in ["infeasible", "unbounded"]:
    raise RuntimeError("*** Error: no stable matrix found")

  Es = X.T @ Me
  n = nfd + 2 * nS
  romAeroS = np.empty((n, n), dtype=np.float)
  romAeroS[0:nfd, 0:nfd] = X.T @ (Ma - margin * Me)
  romAeroS[0:nfd, nfd:(nfd + 2 * nS)] = X.T @ romAero[0:(k + p), nF:(nF + 2 * nS)]
  ids = np.ix_(np.arange(nF, nF + 2 * nS), np.concatenate((np.arange(nfd), np.arange(nF, nF + 2 * nS))))
  romAeroS[nfd:, :] = romAero[ids]
  romAeroS[0:nfd, :] = np.linalg.solve(Es, romAeroS[0:nfd, :])

  if outputAll:
    return romAeroS, X, p, Es
  return romAeroS

def getStabilizedROMdFdX(romAero, nF, nS, nfd, tau, dFdX, margin: float = 1e-8, mu: float = 1e-8, outputAll: bool = False, start: float = None, useMatlab: bool = False, **kwargs):
  romAeroS, X, p, Es = getStabilizedROM(romAero, nF, nS, nfd, tau, margin, mu, True, start, useMatlab, **kwargs)
  dFdXS = dFdX.copy()
  if outputAll:
    return romAeroS, dFdXS, X, p, Es
  return romAeroS, dFdXS

def getStabilizedROMCtrlSurf(romAero, nF, nS, nfd, tau, dFdX, ctrlSurfBlock, margin: float = 1e-8, mu: float = 1e-8, start: float = None, useMatlab: bool = False, **kwargs):
  romAeroS, dFdXS, X, p, Es = getStabilizedROMdFdX(romAero, nF, nS, nfd, tau, dFdX, margin, mu, True, start, useMatlab, **kwargs)
  ids = np.concatenate((np.arange(nfd), np.arange(nF, (nF + 2 * nS))))
  ctrlSurfBlockS = ctrlSurfBlock[ids, :]
  ctrlSurfBlockS[0:nfd, :] = X.T @ ctrlSurfBlockS[0:(nfd + p), :]
  ctrlSurfBlockS[0:nfd, :] = np.linalg.solve(Es, ctrlSurfBlockS[0:nfd, :])

  return romAeroS, dFdXS, ctrlSurfBlockS

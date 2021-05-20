import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
from timeit import default_timer as timer
import pyaeroutils.xpost_utils as xpu
from scipy.linalg import block_diag

try:
  import matlab
  import matlab.engine
  foundMatlab = True
except ModuleNotFoundError:
  foundMatlab = False

def readSV(file: str, isSVD: bool = True):
  if isSVD:
    sv = np.loadtxt(file, skiprows=2)
  else:
    R = np.loadtxt(file, skiprows=1)
    sv = np.sqrt(np.abs(np.diag(R)))
  eng = sv * sv
  reng = np.cumsum(eng / np.sum(eng))

  return sv, eng, reng

def analyzeBasis(file: str, tol: float, verbose: bool = False, plot: bool = False, isSVD: bool = True):
  sv, eng, reng = readSV(file, isSVD)
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

def checkStability(mat, output_largest_real: bool = False):
  ev = np.linalg.eigvals(mat)
  rp = np.real(ev)
  if np.any(rp >= 0.):
    isStable = False
  else:
    isStable = True
  if output_largest_real:
    return isStable, rp.max()
  else:
    return isStable

def checkSubROMStability(romAero, nF, gamma, rhoref, Pref, Vref, output_largest_real: bool = False):
  # Dividing by constant ensures exact same process that occurs in AERO-F occurs here (though really
  # the constant doesn't matter for stability purposes)
  cref = np.sqrt(gamma * Pref / rhoref)
  Mref = Vref / cref
  cnst = Mref * np.sqrt(gamma)
  H = romAero[0:nF, 0:nF] / cnst

  isStable = np.empty((nF, ), dtype=np.bool)
  if output_largest_real:
    largestReal = np.empty((nF, ), dtype=np.float)
    for i in range(nF):
      isStable[i], largestReal[i] = checkStability(H[0:(i + 1), 0:(i + 1)], output_largest_real)
    return isStable, largestReal
  else:
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

def extractOperators(romAero, nF, nS, nFNew = None):
  '''return H, B, C, P, K
  Assumes that no mass matrix inverse has been applied.'''
  if nFNew is None:
    nFNew = nF
  H = -romAero[0:nFNew, 0:nFNew]
  B = -romAero[0:nFNew, nF:(nF + nS)]
  C = -romAero[0:nFNew, (nF + nS):(nF + 2 * nS)]
  P = romAero[nF:(nF + nS), 0:nFNew]
  K = -romAero[nF:(nF + nS), (nF + nS):(nF + 2 * nS)]
  return H, B, C, P, K

def extractOperatorsdFdX(romAero, dFdX, nF, nS, nFNew = None):
  '''return H, B, C, P, K, Py
  Assumes that no mass matrix inverse has been applied.'''
  if nFNew is None:
    nFNew = nF
  H, B, C, P, K = extractOperators(romAero, nF, nS, nFNew)
  if dFdX is not None:
    Py = dFdX.copy()
  else:
    Py = None
  return H, B, C, P, K, Py

def extractOperatorsCtrlSurf(romAero, dFdX, ctrlSurfBlock, nF, nS, nFNew = None):
  '''return H, B, C, P, K, Py, Bcs, Ccs, Pycs
  Assumes that no mass matrix inverse has been applied.'''
  if nFNew is None:
    nFNew = nF
  H, B, C, P, K, Py = extractOperatorsdFdX(romAero, dFdX, nF, nS, nFNew)
  if ctrlSurfBlock is not None:
    nCS = ctrlSurfBlock.shape[1] // 2
    Bcs = -ctrlSurfBlock[0:nFNew, 0:nCS]
    Ccs = -ctrlSurfBlock[0:nFNew, nCS:]
    Pycs = ctrlSurfBlock[nF:(nF + nS), nCS:]
  else:
    Bcs = None
    Ccs = None
    Pycs = None
  return H, B, C, P, K, Py, Bcs, Ccs, Pycs

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
  dF = np.zeros((nT + 1, B.shape[1]), dtype=np.float)
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
    MeM = matlab.double(Me.tolist())
    kM = float(k)
    pM = float(p)
    tauM = float(tau)
    muM = float(mu)

    XM, statusM = eng.stabilizeROM(MaM, MeM, kM, pM, tauM, muM, nargout=2)
    X = np.array(XM._data.tolist())
    X = X.reshape(XM.size, order='F')
    status = str(statusM).lower()
    return X, status

def getStabilizedROM(romAero, nF, nS, nfd, tau, margin: float = 1e-8, mu: float = 1e-8,
                     outputAll: bool = False, start: float = None, useMatlab: bool = False,
                     acceptInaccurate: bool = False, maximum_its: bool = 10, **kwargs):
  if start is None:
    start = timer()
  k = nfd
  P = nF - nfd
  it = 0

  if useMatlab and foundMatlab:
    eng = matlab.engine.start_matlab()
    cdir = eng.cd('/home/users/amcclell/matlab/cvx/')
    eng.cvx_setup(nargout=0)
    odir = eng.cd(cdir)
    successMsgs = ["solved"]
    inaccurateMsgs = ["inaccurate/solved"]
    if acceptInaccurate:
      successMsgs.extend(inaccurateMsgs)
    #eng.run('/home/users/amcclell/matlab/cvx/cvx_startup.m', nargout=0)
  elif useMatlab:
    print('*** Warning: "matlab" module not found. Using cvxpy instead.', flush=True)
    eng = None
    successMsgs = ["optimal"]
    inaccurateMsgs = ["optimal_inaccurate"]
    if acceptInaccurate:
      successMsgs.extend(inaccurateMsgs)
  else:
    eng = None
    successMsgs = ["optimal"]
    inaccurateMsgs = ["optimal_inaccurate"]
    if acceptInaccurate:
      successMsgs.extend(inaccurateMsgs)

  for p in np.arange(1, P + 1):
    Me = np.eye(k + p, k)
    Ma = romAero[0:(k + p), 0:k] + margin * Me

    X, status = stabilizeROM(Ma, Me, k, p, tau, mu, eng, **kwargs)
    print('k + p = {}, status = {}'.format(k + p, status), flush=True)

    if status in successMsgs:
      break
    else:
      print('Problem infeasible/solution tolerance not met. Incrementing p. Elapsed Time: {}'.format(timer() - start))
    
    it += 1
    if it == maximum_its:
      break


  if useMatlab:
    eng.quit()
  
  
  if not acceptInaccurate and status in inaccurateMsgs:
    print('*** Warning: Maximum iterations ({}) reached with only an inaccurate solution'.format(maximum_its))
  elif status not in successMsgs and status not in inaccurateMsgs:
    raise RuntimeError("*** Error: Maximum iterations ({}) reached with no stable matrix".format(maximum_its))

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

def getStabilizedROMdFdX(romAero, nF, nS, nfd, tau, dFdX, margin: float = 1e-8, mu: float = 1e-8,
                         outputAll: bool = False, start: float = None, useMatlab: bool = False,
                         acceptInaccurate: bool = False, maximum_its: bool = 10, **kwargs):
  romAeroS, X, p, Es = getStabilizedROM(romAero, nF, nS, nfd, tau, margin, mu, True, start, useMatlab, acceptInaccurate, maximum_its, **kwargs)
  dFdXS = dFdX.copy()
  if outputAll:
    return romAeroS, dFdXS, X, p, Es
  return romAeroS, dFdXS

def getStabilizedROMCtrlSurf(romAero, nF, nS, nfd, tau, dFdX, ctrlSurfBlock, margin: float = 1e-8,
                             mu: float = 1e-8, start: float = None, useMatlab: bool = False,
                             acceptInaccurate: bool = False, maximum_its: bool = 10, **kwargs):
  romAeroS, dFdXS, X, p, Es = getStabilizedROMdFdX(romAero, nF, nS, nfd, tau, dFdX, margin, mu, True, start, useMatlab, acceptInaccurate, maximum_its, **kwargs)
  ids = np.concatenate((np.arange(nfd), np.arange(nF, (nF + 2 * nS))))
  ctrlSurfBlockS = ctrlSurfBlock[ids, :]
  ctrlSurfBlockS[0:nfd, :] = X.T @ ctrlSurfBlock[0:(nfd + p), :]
  ctrlSurfBlockS[0:nfd, :] = np.linalg.solve(Es, ctrlSurfBlockS[0:nfd, :])

  return romAeroS, dFdXS, ctrlSurfBlockS

def splitSnapshotsByModes(snaps: np.ndarray, nSteps: int, nModesSep: int, sepModes: np.ndarray = None):
  # This assumes frequency domain training where 0 rad/s is included
  sSize = snaps.shape[0]
  nModes = sSize // (2 * nSteps + 1)

  real0 = np.arange(nModes, sSize, 2 * nModes)
  imag0 = np.arange(nModes + 1, sSize, 2 * nModes)

  if sepModes is not None:
    nSeps = sepModes.shape[0]
    snapsSep = np.empty((nSeps, ), dtype=object)

    for i in range(nSeps):
      nM = sepModes[i].shape[0]
      nS = nM * (2 * nSteps + 1)
      
      inds = np.array([], dtype=np.int)
      for j in range(nM):
        m = sepModes[i][j]
        if m >= nModes:
          raise ValueError('Mode ID exceeds number of modes ({} vs {})'.format(m + 1, nModes))
        inds = np.concatenate((inds, np.array([m], dtype=np.int), real0 + 2 * m, imag0 + 2 * m))
      
      inds = np.sort(inds)
      if inds.size != nS:
        raise ValueError('More snapshots requested ({}) than expected ({}) for group {}'.format(inds.size, nS, i))
      
      snapsSep[i] = snaps[inds, :, :]

    return snapsSep
  
  # This is included for backwards compatibility only; the above code is more flexible and does the same thing
  if nModesSep >= nModes:
    raise ValueError('Requested equal number or more modes ({} vs {})'.format(nModesSep, nModes))
  
  nModes2 = nModes - nModesSep
  sSize1 = nModesSep * (2 * nSteps + 1)
  sSize2 = nModes2 * (2 * nSteps + 1)

  i1 = np.arange(nModesSep)

  for i in range(nModesSep):
    i1 = np.concatenate((i1, real0 + 2 * i, imag0 + 2 * i))
  
  i1 = np.sort(i1)
  if i1.size != sSize1:
    raise ValueError('Number of column indices does not match expected size ({} vs {})'.format(i1.size, sSize1))
  i2 = np.setdiff1d(np.arange(sSize), i1, assume_unique=True)

  snaps1 = snaps[i1, :, :]
  snaps2 = snaps[i2, :, :]
  return snaps1, snaps2

def writePODVecs(fN, fNSV, header, rob, sv, dim):
  robXP = xpu.undoReshape(rob, dim)
  robXP = np.concatenate((robXP[0].reshape((1, robXP.shape[1], robXP.shape[2])), robXP), axis=0)
  tags = np.concatenate((np.array([sv.size]), sv * sv))

  xpu.writeXpost(fN, header, tags, robXP)

  with open(fNSV, "w") as f:
    f.write('%d\nSingular values\n' % sv.size)
    for i in range(sv.size):
      f.write('%e ' % sv[i])
    f.write('\n')

def buildDualBasisROMAero(f1: str, f2: str):
  # File 2 is assumed to be the ROM built for the control surfaces
  rom1, dFdX1, ctrlSurf1, nF1, nS1 = readROMAeroCtrlSurf(f1)
  rom2, dFdX2, ctrlSurf2, nF2, nS2 = readROMAeroCtrlSurf(f2)

  if nS1 != nS2:
    raise ValueError('Structural subsystems size mismatch ({} vs {} for ROM 1 and ROM 2'.format(nS1, nS2))
  
  isdFdX = dFdX1 is not None and dFdX2 is not None
  if isdFdX:
    if np.linalg.norm(dFdX1 - dFdX2) > 1e-14:
      raise ValueError('dFdX is not equivalent for ROM 1 and ROM 2 (not affected by fluid reduction, so should be equivalent)')
  
  isCtrlSurf = ctrlSurf1 is not None and ctrlSurf2 is not None
  if isCtrlSurf:
    if np.linalg.norm(ctrlSurf1[nF1:] - ctrlSurf2[nF2:]) > 1e-14:
      raise ValueError('ctrlSurf structural blocks are not equivalent for ROM 1 and ROM 2')

  H1, B1, C1, P1, K1, Py1, Bcs1, Ccs1, Pycs1 = extractOperatorsCtrlSurf(rom1, dFdX1, ctrlSurf1, nF1, nS1)
  H2, B2, C2, P2, K2, Py2, Bcs2, Ccs2, Pycs2 = extractOperatorsCtrlSurf(rom2, dFdX2, ctrlSurf2, nF2, nS2)
  if isCtrlSurf:
    nCS = Pycs2.shape[1]

  if np.linalg.norm(K1 - K2) > 1e-14:
    raise ValueError('Reduced frequencies are not equivalent for ROM 1 and ROM 2')

  rom = np.block([[-H1, np.zeros((nF1, nF2), dtype=np.float), -B1, -C1],
                  [np.zeros((nF2, nF1), dtype=np.float), -H2, np.zeros((nF2, nS1), dtype=np.float), np.zeros((nF2, nS1), dtype=np.float)],
                  [P1, P2, np.zeros((nS1, nS1), dtype=np.float), -K1],
                  [np.zeros((nS1, nF1), dtype=np.float), np.zeros((nS1, nF2), dtype=np.float), np.eye(nS1, dtype=np.float), np.zeros((nS1, nS1), dtype=np.float)]])
  
  if isdFdX:
    dFdX = dFdX1.copy()
  else:
    dFdX = None

  if isCtrlSurf:
    ctrlSurf = np.block([[np.zeros((nF1, 2 * nCS), dtype=np.float)],
                         [-Bcs2, -Ccs2],
                         [np.zeros((nS1, nCS), dtype=np.float), Pycs2],
                         [np.zeros((nS1, 2 * nCS), dtype=np.float)]])
  else:
    ctrlSurf = None
  
  nF = nF1 + nF2
  nS = nS1

  return rom, dFdX, ctrlSurf, nF, nS

def buildMultiBasisROMAero(fNs: np.ndarray, mIDs: np.ndarray, isCS: np.ndarray = None,
                           nFDes: np.ndarray = None):
  '''Combine multiple ROMs in a blocking fashion, i.e. assume that each ROM reduces the following
  equation

  A\dot{w}_i + Hw_i + B_m[:, mID[i]]\dot{y}_i + C_m[:, mID[i]]y_i = 0

  where m = RBM if isCS[i] = False, m = CS otherwise; then, the block ROM is formed by forming the
  following block operators

  A_r = blkdiag(A_{r,i}), H_r = blkdiag(H_{r,i}), B_r = blkdiag(B_{r,i}), C_r = blkdiag(C_{r,i})
  B_{r,CS} = blkdiag(B_{r,CS,i}), C_{r,CS} = blkdiag(C_{r,CS,i})

  where A_{r,i} is identity, H_{r,i} is the reduced fluid Jacobian for each ROM, B_{r,i} = B_{r,CS,i} = 0,
  C_{r,i} = C_{r,CS,i} = 0 for all columns that correspond to modes not including in the training for basis
  i (i.e. for all modes, matrices not included in mIDs[i] and specified by isCS[i])

  If nFDes is not None, the entries specify the desired size of the fluid subsystem.

  '''
  nFiles = fNs.shape[0]
  nMG = mIDs.shape[0]
  if nFiles != nMG:
    raise ValueError('Number of file names ({}) does not equal number of mode ID groups ({})'.format(nFiles, nMG))
  if isCS is not None:
    nICS = isCS.shape[0]
    if nMG != nICS:
      raise ValueError('Number of mode ID groups ({}) does not equal number of RBM/CS identifiers ({})'.format(nMG, nICS))
  else:
    nICS = np.zeros((nMG, ), dtype=bool)
  if nFDes is not None:
    nnF = nFDes.shape[0]
    if nnF != nFiles:
      raise ValueError('Number of desired fluid ROM sizes ({}) does not equal number of file names ({})'.format(nnF, nFiles))
  else:
    nFDes = np.full((nFiles, ), None)

  roms = np.empty((nFiles, ), dtype=object)
  dFdXs = np.empty_like(roms)
  ctrlSurfs = np.empty_like(roms)
  nFs = np.empty((nFiles, ), dtype=np.int)

  Hs = np.empty_like(roms)
  Bs = np.empty_like(roms)
  Cs = np.empty_like(roms)
  Ps = np.empty_like(roms)
  Ks = np.empty_like(roms)
  Pys = np.empty_like(roms)
  Bcss = np.empty_like(roms)
  Ccss = np.empty_like(roms)
  Pycss = np.empty_like(roms)

  roms[0], dFdXs[0], ctrlSurfs[0], nFs[0], nS = readROMAeroCtrlSurf(fNs[0])
  if nFDes[0] is not None:
    if nFDes[0] > nFs[0]:
      print('*** Warning: desired fluid ROM size ({}) larger than maximum size ({}) for ROM 0. Choosing maximum size.'.format(nFDes[0], nFs[0]))
      nFDes[0] = nFs[0]
  Hs[0], Bs[0], Cs[0], Ps[0], Ks[0], Pys[0], Bcss[0], Ccss[0], Pycss[0] = extractOperatorsCtrlSurf(roms[0], dFdXs[0], ctrlSurfs[0], nFs[0], nS, nFDes[0])

  if dFdXs[0] is not None:
    isdFdX = True
  else:
    isdFdX = False
  
  if ctrlSurfs[0] is not None:
    nCS = ctrlSurfs[0].shape[1] // 2
  else:
    nCS = 0
  
  mNS = np.arange(nS)
  mCS = np.arange(nCS)

  if not isCS[0]:
    im = np.setdiff1d(mNS, mIDs[0])
    Bs[0][:, im] = np.zeros((Bs[0].shape[0], im.size), dtype=np.float)
    Cs[0][:, im] = np.zeros((Cs[0].shape[0], im.size), dtype=np.float)
    Bcss[0] = np.zeros((Hs[0].shape[0], nCS), dtype=np.float)
    Ccss[0] = np.zeros((Hs[0].shape[0], nCS), dtype=np.float)
  elif nCS != 0 and isCS[0]:
    im = np.setdiff1d(mCS, mIDs[0])
    Bcss[0][:, im] = np.zeros((Bcss[0].shape[0], im.size), dtype=np.float)
    Ccss[0][:, im] = np.zeros((Ccss[0].shape[0], im.size), dtype=np.float)
    Bs[0] = np.zeros((Hs[0].shape[0], nS), dtype=np.float)
    Cs[0] = np.zeros((Hs[0].shape[0], nS), dtype=np.float)
  else:
    raise ValueError('Mode Group 0 is set to "Control Surface" but there are no control surfaces')

  for i in np.arange(1, nFiles):
    roms[i], dFdXs[i], ctrlSurfs[i], nFs[i], tmpnS = readROMAeroCtrlSurf(fNs[i])

    if tmpnS != nS:
      raise ValueError('Number of structural modes in {} ({}) do not match number in {} ({})'.format(fNs[i], tmpnS, fNs[0], nS))
    
    if isdFdX and dFdXs[i] is not None:
      if np.linalg.norm(dFdXs[i] - dFdXs[0]) > 1e-14:
        raise ValueError('dFdX is not equivalent for ROM {} and ROM {}'.format(fNs[i], fNs[0]))
    
    if ctrlSurfs[i] is not None:
      tmpnCS = ctrlSurfs[i].shape[1] // 2
      if nCS != tmpnCS:
        raise ValueError('ROM {} has {} control surfaces, but ROM {} has {}'.format(fNs[i], tmpnCS, fNs[0], nCS))
    elif nCS != 0:
      raise ValueError('ROM {} has 0 control surfaces, but ROM {} has {}'.format(fNs[i], fNs[0], nCS))

    if nFDes[i] is not None:
      if nFDes[i] > nFs[i]:
        print('*** Warning: desired fluid ROM size ({}) larger than maximum size ({}) for ROM {}. Choosing maximum size.'.format(nFDes[i], nFs[i], i))
        nFDes[i] = nFs[i]
    Hs[i], Bs[i], Cs[i], Ps[i], Ks[i], Pys[i], Bcss[i], Ccss[i], Pycss[i] = extractOperatorsCtrlSurf(roms[i], dFdXs[i], ctrlSurfs[i], nFs[i], tmpnS, nFDes[i])

    if nS != 0:
      if np.linalg.norm(Ks[i] - Ks[0]) > 1e-14:
        raise ValueError('K is not equivalent for ROM {} and ROM {}'.format(fNs[i], fNs[0]))
    
    if nCS != 0:
      if np.linalg.norm(Pycss[i] - Pycss[0]) > 1e-14:
        raise ValueError('Pycs is not equivalent for ROM {} and ROM {}'.format(fNs[i], fNs[0]))

    if not isCS[i]:
      im = np.setdiff1d(mNS, mIDs[i])
      Bs[i][:, im] = np.zeros((Bs[i].shape[0], im.size), dtype=np.float)
      Cs[i][:, im] = np.zeros((Cs[i].shape[0], im.size), dtype=np.float)
      Bcss[i] = np.zeros((Hs[i].shape[0], nCS), dtype=np.float)
      Ccss[i] = np.zeros((Hs[i].shape[0], nCS), dtype=np.float)
    elif nCS != 0 and isCS[i]:
      im = np.setdiff1d(mCS, mIDs[i])
      Bcss[i][:, im] = np.zeros((Bcss[i].shape[0], im.size), dtype=np.float)
      Ccss[i][:, im] = np.zeros((Ccss[i].shape[0], im.size), dtype=np.float)
      Bs[i] = np.zeros((Hs[i].shape[0], nS), dtype=np.float)
      Cs[i] = np.zeros((Hs[i].shape[0], nS), dtype=np.float)
    else:
      raise ValueError('Mode Group {} is set to "Control Surface" but there are no control surfaces'.format(i))
    
  H = block_diag(*Hs.tolist())
  B = np.block(Bs.reshape((nFiles, 1)).tolist())
  C = np.block(Cs.reshape((nFiles, 1)).tolist())
  P = np.block(Ps.tolist())
  Bcs = np.block(Bcss.reshape((nFiles, 1)).tolist())
  Ccs = np.block(Ccss.reshape((nFiles, 1)).tolist())

  nF = H.shape[1]
  zNS = np.zeros((nS, nS), dtype=np.float)
  zW = np.zeros((nS, nF), dtype=np.float)
  iNS = np.eye(nS, dtype=np.float)
  rom = np.block([[-H, -B, -C],
                  [P, zNS, -Ks[0]],
                  [zW, iNS, zNS]])
  
  dFdX = Pys[0]

  if nCS != 0:
    zCS = np.zeros((nS, nCS), dtype=np.float)
    ctrlSurf = np.block([[-Bcs, -Ccs],
                         [zCS, Pycss[0]],
                         [zCS, zCS]])
  else:
    ctrlSurf = None
  
  return rom, dFdX, ctrlSurf, nF, nS

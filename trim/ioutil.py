import numpy as np
import subprocess
import sys
from io import TextIOWrapper


def safeCall(cmd: str):
  output = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
  if output.returncode:
    print('*** Non-zero exit code from "{}"'.format(cmd))
    exit(output.returncode)
  return output


def readLastLine(file: str):
  output = subprocess.run('tail -n 1 {}'.format(file), shell=True, stdout=subprocess.PIPE)
  if output.returncode:
    print('*** Non-zero exit code from "tail -n 1 {}"'.format(file))
    exit(output.returncode)
  return output.stdout.decode().strip()


def getSteadyForce(file: str):
  line = readLastLine(file)
  F = np.fromstring(line, sep=' ')
  F = F[4:10]
  return F


def getSteadyLifts(file: str):
  line = readLastLine(file)
  L = np.fromstring(line, sep=' ')
  L = L[-3:]
  return L


def getSteadySensitivitiesOfLifts(file: str):
  dFdS = np.atleast_2d(np.loadtxt(file, skiprows=1))
  iAoA = (dFdS[:, 1] == 3)
  iDX = (dFdS[:, 1] == 1)
  dLdA = dFdS[iAoA, -3:].squeeze()
  dLdX = dFdS[iDX, -3:]
  return dLdA, dLdX


def getSteadySensitivitiesOfForces(file: str):
  dFdS = np.atleast_2d(np.loadtxt(file, skiprows=1))
  iAoA = (dFdS[:, 1] == 3)
  iDX = (dFdS[:, 1] == 1)
  dFdA = dFdS[iAoA, -7:-1].squeeze()
  dFdX = dFdS[iDX, -7:-1]
  return dFdA, dFdX


def copyAndRename(origin: str, name: str, ext: str, destination: str, idO: int, idD: int):
  idOs = ''
  exts = ''
  if idO != -1:
    idOs = '{}.'.format(idO)
    exts = '{}\\.{}'.format(idO, ext)
  else:
    exts = ext
  cmd0 = 'find {} -type f -name "{}.{}{}*" -execdir'.format(origin, name, idOs, ext)
  cmd = '{} bash -c \'var=${{0:2}} && pwd && cp -v $0 {}${{var/{}/{}.{}}}\' {{}} \\;'.format(cmd0, destination, exts, idD, ext)
  output = safeCall(cmd)
  return output


def writeTrimIteration(f: TextIOWrapper, it: int, alt: float, AoA: float, Tf: float, u: np.ndarray, dLdA: np.ndarray, dLdu: np.ndarray):
  nCtrlSurf = u.shape[0]
  f.write('{:d} {:.16e} {:.16e} {:.16e} {:d}'.format(it, alt, AoA, Tf, nCtrlSurf))
  for cs in u:
    f.write(' {:.16e}'.format(cs))
  for s in dLdA:
    f.write(' {:.16e}'.format(s))
  for i in range(nCtrlSurf):
    for s in dLdu[i]:
      f.write(' {:.16e}'.format(s))
  f.write('\n')


def writeTrim(file: str, it: int, alt: float, AoA: float, Tf: float, u: np.ndarray, dLdA: np.ndarray, dLdu: np.ndarray):
  if(it == 0):
    with open(file, 'w') as f:
      f.write('ref. frame: x=freestream direction, y=inertial y (no sideslip), z=perpindicular to both x, y\n')
      f.write('Iteration Altitude AoA Thrust nCtrlSurf')
      nCtrlSurf = u.shape[0]
      for i in range(nCtrlSurf):
        f.write(' cs{:d}'.format(i))
      f.write(' dLxdAoA dLydAoA dLzdAoA dMxdAoA dMydAoA dMzdAoA')
      for i in range(nCtrlSurf):
        f.write(' dLxdcs{id:d} dLydcs{id:d} dLzdcs{id:d} dMxdcs{id:d} dMydcs{id:d} dMzdcs{id:d}'.format(id=i))
      f.write('\n')
      writeTrimIteration(f, it, alt, AoA, Tf, u, dLdA, dLdu)
  else:
    with open(file, 'a') as f:
      writeTrimIteration(f, it, alt, AoA, Tf, u, dLdA, dLdu)


def makeDir(path: str):
  cmd = "mkdir -p {}".format(path)
  output = safeCall(cmd)
  return output


class Tee(object):
  def __init__(self, file: str):
    self.terminal = sys.stdout
    self.log = open(file, 'w')

  def write(self, s):
    self.terminal.write(s)
    self.log.write(s)

  def flush(self):
    self.terminal.flush()
    self.log.flush()

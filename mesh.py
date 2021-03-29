import pyaeroutils.xpost_utils as xpost_utils
import numpy as np

def computeAverageWallDistance(mesh: str, dwall: str, volID: int = 0):
  with open(dwall) as f:
    data = f.read().splitlines()
  data[0:3] = []
  data = np.asarray(data, dtype=np.float64)

  nodes, elements, elementNames, nElements, nNodesPerElement = xpost_utils.readMesh(mesh)

  els = elements[volID]
  els = els[:, 2:6]

  wall = data == 0.0
  inds = np.arange(0, data.shape[0])
  inds = inds[wall]

  layer1 = np.unique(np.nonzero(np.isin(els, inds))[0])
  layer1 = np.unique(els[layer1, :])
  layer1 = layer1[np.isin(layer1, inds, invert=True)]

  yp = np.mean(data[layer1])
  return yp

def fixHalfMesh(mesh: str, symPlaneID: int, coordID: int, coord: float = 0.0, verbose: bool = False):
  start = xpost_utils.timer()

  if verbose:
    print('Reading {} ...'.format(mesh), flush=True)
  nodes, elements, elementNames, nElements, nNodesPerElement = xpost_utils.readMesh(mesh, start, verbose=verbose)
  end = timer()
  if verbose:
    print('Done reading {}.    Elapsed Time: {}'.format(mesh, end - start), flush=True)
    print('Fixing symmetry plane ... ', flush=True)
  
  nIDs = np.unique(elements[symPlaneID])
  inds = np.abs(nodes[nIDs, coordID] - coord) > 1e-10

  if verbose:
    print('Found {} nodes on Symmetry surface at |x_{}| > 1e-10'.format(np.sum(inds), coordID), flush=True)
  nodes[nIDs[inds], 1] = 0.

  end = timer()
  if verbose:
    print('Fixed symmetry plane.    Elapsed Time: {}'.format(end - start), flush=True)
    print('Writing adjusted mesh ... ', end='', flush=True)
  writeMesh(filename, nodes, elements, elementNames, nElements, nNodesPerElement, shared)
  end = timer()
  if verbose:
    print('Done.    Elapsed Time: {}'.format(end - start), flush=True)

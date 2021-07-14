import numpy as np
import subprocess
import ctypes
import multiprocessing as mp
from timeit import default_timer as timer
from itertools import islice

def getLabelLineNo(filename: str):
  output = subprocess.run('grep -n "Elements" {} | awk \'{{print $1}}\' | cut -d\':\' -f1'.format(filename), shell=True, stdout=subprocess.PIPE)
  if output.returncode:
    print('*** Non-zero exit code in getLabelLineNo for file "{}"'.format(filename))
    exit(output.returncode)
  output2 = subprocess.run('wc -l {} | awk \'{{print $1}}\''.format(filename), shell=True, stdout=subprocess.PIPE)
  if output2.returncode:
    print('*** Non-zero exit code in getLabelLineNo for file "{}"'.format(filename))
    exit(output2.returncode)
  lineNo = np.fromstring(output.stdout.decode() + output2.stdout.decode().strip(), dtype=np.int, sep='\n')
  lineNo[-1] += 1
  return lineNo

def readMesh(filename: str, start: float = None, shared: bool = False, verbose: bool = False):
  # labelNo = getLabelLineNo(filename)
  # nGroups = labelNo.shape[0] - 1

  # elements = np.empty((nGroups,), dtype=object)

  # with open(filename) as f:
  #   lines = np.array(f.readlines())
  #   nodes = np.genfromtxt(lines[1:labelNo[0] - 1], dtype=np.float64, usecols=(1, 2, 3))

  #   elementNames = np.char.strip(lines[labelNo[0:-1] - 1])

  #   for i in range(nGroups):
  #     elements[i] = np.genfromtxt(lines[labelNo[i]:labelNo[i + 1] - 1], dtype=np.int)[:, 2:]
  #     elements[i] -= 1

  # print('Read {}'.format(filename))
  # print('Found {} nodes'.format(nodes.shape[0]))
  # for i in range(elements.shape[0]):
  #   print('Found "{}" with {} {}-node elements'.format(elementNames[i], elements[i].shape[0], elements[i].shape[1]))
  # return nodes, elements, elementNames
  if start is None:
    start = timer()
  labelNo = getLabelLineNo(filename)
  nGroups = labelNo.shape[0] - 1

  elements = np.empty((nGroups,), dtype=object)

  with open(filename) as f:
    header = f.readline()
    nNodes = int(labelNo[0] - 2)
    if shared:
      nodes = mp.RawArray(ctypes.c_double, nNodes * 3)
      nodesNP = np.frombuffer(nodes).reshape((nNodes, 3))
    else:
      nodes = np.empty((nNodes, 3), dtype=np.float64)
      nodesNP = nodes
    for i in range(nNodes):
      line = f.readline()
      nodesNP[i] = np.fromstring(line, dtype=np.float64, sep=' ')[1:]
      if (i + 1) % 500000 == 0:
        end = timer()
        if verbose:
          print('Read {} nodes ... Elapsed Time: {}'.format(i + 1, end - start), flush=True)
    if verbose:
      print('Found {} nodes'.format(nNodes), flush=True)

    elementNames = np.empty((nGroups, ), dtype=object)
    if shared:
      nNodesPerElement = mp.RawArray(ctypes.c_long, nGroups)
      nNPElNP = np.frombuffer(nNodesPerElement, dtype=np.int64)

      nElements = mp.RawArray(ctypes.c_long, nGroups)
      nElsNP = np.frombuffer(nElements, dtype=np.int64)
    else:
      nNodesPerElement = np.empty((nGroups, ), dtype=np.int64)
      nNPElNP = nNodesPerElement

      nElements = np.empty((nGroups, ), dtype=np.int64)
      nElsNP = nElements

    for i in range(nGroups):
      nEls = int(labelNo[i + 1] - 1 - labelNo[i])
      nElsNP[i] = nEls
      line = f.readline()
      elementNames[i] = line.strip()
      line = f.readline()
      tmp = np.fromstring(line, dtype=np.int32, sep=' ')[1:]
      eType = tmp[0]
      if eType == 4:
        nNodesEl = 3
      elif eType == 5:
        nNodesEl = 4
      else:
        print('*** Error: Unknown element type {}'.format(eType), flush=True)
        exit(-1)
      nNPElNP[i] = nNodesEl
      if shared:
        elements[i] = mp.RawArray(ctypes.c_long, nEls * nNodesEl)
        eiNP = np.frombuffer(elements[i], dtype=np.int64).reshape((nEls, nNodesEl))
      else:
        elements[i] = np.empty((nEls, nNodesEl), dtype=np.int64)
        eiNP = elements[i]
      eiNP[0] = tmp[1:]
      if verbose:
        print('Found "{}" with {} {}-node elements'.format(elementNames[i], eiNP.shape[0], eiNP.shape[1]), flush=True)
      for j in np.arange(1, nEls):
        line = f.readline()
        eiNP[j] = np.fromstring(line, dtype=np.int64, sep=' ')[2:]
        if (j + 1) % 500000 == 0:
          end = timer()
          if verbose:
            print('Read {} elements from group {} ... Elapsed Time: {}'.format(j + 1, i + 1, end - start), flush=True)
      eiNP -= 1

  return nodes, elements, elementNames, nElements, nNodesPerElement

def writeMesh(filename: str, nodes, elements, elementNames, nElements, nNodesPerElement, shared: bool = False):
  if shared:
    nodesNP = np.frombuffer(nodes, dtype=np.float64).reshape((len(nodes) // 3, 3))
    elementsNP = np.empty_like(elements, dtype=object)
    for i in range(elements.shape[0]):
      elementsNP[i] = np.frombuffer(elements[i], dtype=np.int64).reshape((np.int64(nElements[i]), np.int64(nNodesPerElement[i])))
  else:
    nodesNP = nodes
    elementsNP = elements
  
  with open(filename, "w") as f:
    f.write('Nodes FluidNodes\n')
    for i in range(nodesNP.shape[0]):
      f.write('%d' % (i + 1))
      f.write(' % .16e % .16e % .16e\n' % tuple(nodesNP[i]))
    for i in range(elementNames.shape[0]):
      f.write('%s\n' % elementNames[i])
      if elementsNP[i].shape[1] == 4:
        t = 5
        nNPE = 4
      elif elementsNP[i].shape[1] == 3:
        t = 4
        nNPE = 3
      else:
        print('*** Error: wrong element type...', flush=True)
        exit(-1)
      for j in range(elementsNP[i].shape[0]):
        f.write('%d %d' % (j + 1, t))
        for k in range(nNPE):
          f.write(' %d' % (elementsNP[i][j, k] + 1))
        f.write('\n')

def readXpostCore(lines: np.ndarray, nVectors: int, nNodes: int, start: float = None, shared: bool = False):
  if start is None:
    start = timer()
  nComp = np.atleast_1d(np.genfromtxt([lines[1]], dtype=np.float64)).shape[0]
  
  i = 1
  j = i + nNodes

  if shared:
    output = mp.RawArray(ctypes.c_double, nVectors * nNodes * nComp)
    outputNP = np.frombuffer(output).reshape((nVectors, nNodes, nComp))

    tags = mp.RawArray(ctypes.c_double, nVectors)
    tagsNP = np.frombuffer(tags)
  else:
    output = np.empty((nVectors, nNodes, nComp), dtype=np.float64)
    outputNP = output.view()

    tags = np.empty((nVectors, ), dtype=np.float64)
    tagsNP = tags.view()

  for k in range(nVectors):
    tagsNP[k] = np.float64(lines[i - 1])
    outputNP[k] = np.genfromtxt(lines[i:j], dtype=np.float64).reshape((nNodes, nComp))
    i = j + 1
    j = i + nNodes
    end = timer()
    print('Stored vector {}    Elapsed Time: {}'.format(k + 1, end - start), flush=True)
  
  return tags, output

def readXpost(filename: str, start: float = None, shared: bool = False):
  if start is None:
    start = timer()
  
  with open(filename) as f:
    lines = np.array(f.readlines())
    end = timer()
    print('Read all lines from {}    Elapsed Time: {}'.format(filename, end - start), flush=True)
    header = lines[0].strip()
    nNodes = np.int(lines[1].strip())
    nLines = lines.shape[0]
    nVectors = nLines // nNodes
    nComp = np.atleast_1d(np.genfromtxt([lines[3]], dtype=np.float64)).shape[0]

  #   i = 3
  #   j = i + nNodes
    
  #   if shared:
  #     output = mp.RawArray(ctypes.c_double, nVectors * nNodes * nComp)
  #     outputNP = np.frombuffer(output).reshape((nVectors, nNodes, nComp))

  #     tags = mp.RawArray(ctypes.c_double, nVectors)
  #     tagsNP = np.frombuffer(tags)
  #   else:
  #     output = np.empty((nVectors, nNodes, nComp), dtype=np.float64)
  #     outputNP = output.view()

  #     tags = np.empty((nVectors, ), dtype=np.float64)
  #     tagsNP = tags.view()

  #   for k in range(nVectors):
  #     tagsNP[k] = np.float64(lines[i - 1])
  #     outputNP[k] = np.genfromtxt(lines[i:j], dtype=np.float64).reshape((nNodes, nComp))
  #     i = j + 1
  #     j = i + nNodes
    
  # return header, tags, output

    tags, output = readXpostCore(lines[2:], nVectors, nNodes, start, shared)
  return header, tags, output

def readXpostSubset(filename: str, nVectors: int, startVector: int = 0, start: float = None, shared: bool = False):
  if start is None:
    start = timer()
  
  with open(filename) as f:
    header = f.readline().strip()
    nNodes = np.int(f.readline().strip())
    startID = startVector * (nNodes + 1)
    nLines = nVectors * (nNodes + 1)
    lines = np.array(list(islice(f, startID, startID + nLines)))
    end = timer()
    print('Read {} lines from {}    Elapsed Time: {}'.format(nLines, filename, end - start), flush=True)

    tags, output = readXpostCore(lines, nVectors, nNodes, start, shared)
  return header, tags, output

def reshapeXpost(input: np.ndarray):
  nV, nN, nC = input.shape
  return input.reshape((nV, nN * nC), order='C')

def undoReshape(input: np.ndarray, nC: int):
  nV, nT = input.shape
  nN = nT // nC
  if nN - (nT / nC) != 0:
    raise ValueError('Number of components ({}) does not lead to an integer number of nodes ({})'.format(nC, nT / nC))
  return input.reshape((nV, nN, nC), order='C')

def writeXpost(filename: str, header, tags, output):
  with open(filename, "w") as f:
    f.write('%s\n' % header)
    f.write('%d\n' % output.shape[1])

    for i in range(output.shape[0]):
      f.write('%.16e\n' % tags[i])
      for j in range(output.shape[1]):
        vec = output[i, j]
        for v in vec:
          f.write(' % .16e' % v)
        f.write('\n')

def readGeneralizedOutput(filename: str):
  with open(filename) as f:
    lines = np.array(f.readlines())
    header = lines[0].strip()
    nComp = np.int(lines[1].strip())
    nLines = lines.shape[0]
    nSteps = nLines // 2 - 1

    i = 2
    j = i + 1

    tags = np.empty((nSteps, ), dtype=np.float64)
    output = np.empty((nSteps, nComp), dtype=np.float64)

    for k in range(nSteps):
      tags[k] = np.float64(lines[i])
      output[k] = np.genfromtxt([lines[j]], dtype=np.float64)
      i += 2
      j += 2

  return tags, output

def removeExt(filename: str, ext: str = '.xpost'):
  i = filename.index(ext)
  return filename[0:i]

def splitTimesteps(filename: str, ext: str = '.xpost'):
  header, tags, output = readXpost(filename)

  if output.shape[2] > 1:
    s1 = 'Vector '
  else:
    s1 = 'Scalar '
  s2 = 'under'

  i0 = header.index(s1) + len(s1)
  i1 = header.index(s2) - 1

  name = header[i0:i1]
  tail = header[i1:]

  fn = removeExt(filename, ext)

  for i in range(tags.size):
    fni = '{}.{}{}'.format(fn, i, ext)
    hi = '{}{}.{}{}'.format(s1, name, i, tail)
    writeXpost(fni, hi, np.atleast_1d(tags[i]), np.expand_dims(output[i], 0))

def readForce(filename: str):
  forceAll = np.loadtxt(filename, skiprows=1, dtype=np.float)
  t = forceAll[:,1]
  force = forceAll[:,4:-1]
  return t, force

def readLiftAndDrag(filename: str):
  tmp = np.loadtxt(filename, skiprows=1, dtype=np.float)
  t = tmp[:,1]
  liftAndDrag = tmp[:,4:-1]
  return t, liftAndDrag

def readGenericTopFile(filename: str, start: float = None, verbose: bool = False):
  if start is None:
    start = timer()
  
  labelNo = getLabelLineNo(filename)
  elements = dict()

  with open(filename) as f:
    lines = f.readlines()
  end = timer()
  if verbose:
    print('Read all lines from {} ... Elapsed Time: {}'.format(filename, end - start), flush=True)

  nNodes = int(labelNo[0] - 2)
  nodes = np.empty((nNodes, 3), dtype=np.float64)
  j = 1
  for i in range(nNodes):
    line = lines[j]
    j += 1
    nodes[i] = np.fromstring(line, dtype=np.float64, sep=' ')[1:]
    if (i + 1) % 500000 == 0:
      end = timer()
      if verbose:
        print('Read {} nodes ... Elapsed Time: {}'.format(i + 1, end - start), flush=True)
  if verbose:
    print('Found {} nodes'.format(nNodes), flush=True)

  cnt = 0
  for k in np.arange(j, len(lines)):
    line = lines[k]
    if 'Elements' in line:
      continue
    tmp = np.fromstring(line, dtype=int, sep=' ')[2:]
    nNpEl = tmp.size
    if nNpEl not in elements:
      elements[nNpEl] = np.empty((0, nNpEl), dtype=int)
    elements[nNpEl] = np.vstack((elements[nNpEl], tmp))

  for key in iter(elements):
    elements[key] -= 1
  
  nNodesPerElement = np.array(list(elements.keys()), dtype=int)
  elements = np.array(list(elements.values()))
  return nodes, elements, nNodesPerElement

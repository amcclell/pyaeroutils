import numpy as np
import matplotlib.pyplot as plt

def plotLong(t: np.ndarray, x: np.ndarray, v: np.ndarray, titles: np.ndarray = None,
             xlabels: np.ndarray = None, ylabels: np.ndarray = None, lineType: str = '-',
             legends: np.ndarray = None, axes: np.ndarray = None):
  """Plot longitudinal postion and velocity (translations and rotations).

  Parameters
  ----------
  t: numpy-array
     Array of time values (shape = (n, ))
  x: numpy-array
     Array of longitudinal displacements (shape = (n, 3)), in x, z, theta_y order.
  v: numpy-array
     Array of longitudinal velocities (shape = (n, 3)), in same order as above
  titles: numpy-array (default = None)
     Array of plot titles (shape = (4, )), in displacement, velocity, rotation and
     rotation rate order. If None, default titles are used.
  xlabels: numpy-array (default = None)
     Array of plot x-axis labels (shape = (4, )), in same order as for title. If
     None, default labels are used.
  ylabels: numpy-array (default = None)
     Array of plot y-axis labels (shape = (4, )), in same order as for title. If
     None, default labels are used.
  lineType: string (default = '-')
     String containing the line dash type ('-', '.', '--', '-.')
  legends: numpy-array (default = None)
     Array of line labels (shape = (6, )) for x-displacement, z-displacement, their
     velocities, y-rotation, its rotation rate. If None, default labels are used.
  axes: numpy-array (default = None)
     Array of matplotlib.pyplot axes (shape = (4, )) to be used for plotting. If None,
     new axes are generated.
  
  Returns
  -------
  fig: matplotlib.pyplot figure
  ax1, ax2, ax3, ax4: matplotlib.pyplot axes (for plots in order listed for titles)
  """
  if axes is None:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
  else:
    ax1, ax2, ax3, ax4 = tuple(axes)
    fig = ax1.get_figure()

  if titles is None:
    titles = np.array([r'Longitudinal Position', r'Longitudinal Velocity',
                       r'Longitudinal Rotation', r'Longitudinal Rotation Rate'])
  if xlabels is None:
    xlabels = np.array([r'Time, (s)', r'Time, (s)',
                        r'Time, (s)', r'Time, (s)'])
  
  if ylabels is None:
    ylabels = np.array([r'Position Error, (m)', r'Velocity Error, (m)',
                        r'Rotation Error, (rad)', r'Convected Rotation Rate Error, (rad / s)'])
  
  if legends is None:
    legends = np.array([r'$\delta x$', r'$\delta z$',
                        r'$\delta\dot{x}$', r'$\delta\dot{z}$',
                        r'$\delta\theta_y$', r'$\delta\dot{\theta}_y$'])

  ax1.plot(t, x[:, 0], lineType + 'b', label=legends[0])
  ax1.plot(t, x[:, 1], lineType + 'm', label=legends[1])
  ax1.set_xlabel(xlabels[0])
  ax1.set_ylabel(ylabels[0])
  ax1.set_title(titles[0])
  ax1.legend()

  ax2.plot(t, v[:, 0], lineType + 'b', label=legends[2])
  ax2.plot(t, v[:, 1], lineType + 'm', label=legends[3])
  ax2.set_xlabel(xlabels[1])
  ax2.set_ylabel(ylabels[1])
  ax2.set_title(titles[1])
  ax2.legend()

  ax3.plot(t, x[:, 2], lineType + 'g', label=legends[4])
  ax3.set_xlabel(xlabels[2])
  ax3.set_ylabel(ylabels[2])
  ax3.set_title(titles[2])
  ax3.legend()

  ax4.plot(t, v[:, 2], lineType + 'g', label=legends[5])
  ax4.set_xlabel(xlabels[3])
  ax4.set_ylabel(ylabels[3])
  ax4.set_title(titles[3])
  ax4.legend()

  return fig, ax1, ax2, ax3, ax4

def plotLat(t: np.ndarray, x: np.ndarray, v: np.ndarray, titles: np.ndarray = None,
             xlabels: np.ndarray = None, ylabels: np.ndarray = None, lineType: str = '-',
             legends: np.ndarray = None, axes: np.ndarray = None):
  """Plot lateral postion and velocity (translations and rotations).

  Parameters
  ----------
  t: numpy-array
     Array of time values (shape = (n, ))
  x: numpy-array
     Array of lateral displacements (shape = (n, 3)), in y, theta_x, theta_z
     order.
  v: numpy-array
     Array of lateral velocities (shape = (n, 3)), in same order as above
  titles: numpy-array (default = None)
     Array of plot titles (shape = (4, )), in displacement, velocity, rotation and
     rotation rate order. If None, default titles are used.
  xlabels: numpy-array (default = None)
     Array of plot x-axis labels (shape = (4, )), in same order as for title. If
     None, default labels are used.
  ylabels: numpy-array (default = None)
     Array of plot y-axis labels (shape = (4, )), in same order as for title. If
     None, default labels are used.
  lineType: string (default = '-')
     String containing the line dash type ('-', '.', '--', '-.')
  legends: numpy-array (default = None)
     Array of line labels (shape = (6, )) for y-displacement, its velocity,
     x-rotation, z-rotation, and their rotation rates. If None, default labels
     are used.
  axes: numpy-array (default = None)
     Array of matplotlib.pyplot axes (shape = (4, )) to be used for plotting. If None,
     new axes are generated.
  
  Returns
  -------
  fig: matplotlib.pyplot figure
  ax1, ax2, ax3, ax4: matplotlib.pyplot axes (for plots in order listed for titles)
  """
  if axes is None:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
  else:
    ax1, ax2, ax3, ax4 = tuple(axes)
    fig = ax1.get_figure()

  if titles is None:
    titles = np.array([r'Lateral Position', r'Lateral Velocity',
                       r'Lateral Rotation', r'Lateral Rotation Rate'])
  if xlabels is None:
    xlabels = np.array([r'Time, (s)', r'Time, (s)',
                        r'Time, (s)', r'Time, (s)'])
  
  if ylabels is None:
    ylabels = np.array([r'Position Error, (m)', r'Velocity Error, (m)',
                        r'Rotation Error, (rad)', r'Convected Rotation Rate Error, (rad / s)'])
  
  if legends is None:
    legends = np.array([r'$\delta y$', r'$\delta\dot{y}$',
                        r'$\delta\theta_x$', r'$\delta\theta_z$',
                        r'$\delta\dot{\theta}_x$', r'$\delta\dot{\theta}_z$'])

  ax1.plot(t, x[:, 0], lineType + 'g', label=legends[0])
  ax1.set_xlabel(xlabels[0])
  ax1.set_ylabel(ylabels[0])
  ax1.set_title(titles[0])
  ax1.legend()

  ax2.plot(t, v[:, 0], lineType + 'g', label=legends[1])
  ax2.set_xlabel(xlabels[1])
  ax2.set_ylabel(ylabels[1])
  ax2.set_title(titles[1])
  ax2.legend()

  ax3.plot(t, x[:, 0], lineType + 'b', label=legends[2])
  ax3.plot(t, x[:, 1], lineType + 'm', label=legends[3])
  ax3.set_xlabel(xlabels[2])
  ax3.set_ylabel(ylabels[2])
  ax3.set_title(titles[2])
  ax3.legend()

  ax4.plot(t, v[:, 0], lineType + 'b', label=legends[4])
  ax4.plot(t, v[:, 1], lineType + 'm', label=legends[5])
  ax4.set_xlabel(xlabels[3])
  ax4.set_ylabel(ylabels[3])
  ax4.set_title(titles[3])
  ax4.legend()

  return fig, ax1, ax2, ax3, ax4

def plotLatLong(t: np.ndarray, x: np.ndarray, v: np.ndarray, titles: np.ndarray = None,
                xlabels: np.ndarray = None, ylabels: np.ndarray = None, lineType: str = '-',
                legends: np.ndarray = None, axes: np.ndarray = None):
  """Plot lateral and longitudinal postion and velocity (translations and rotations).

  Parameters
  ----------
  t: numpy-array
     Array of time values (shape = (n, ))
  x: numpy-array
     Array of lateral displacements (shape = (n, 6)), in x, y, z, theta_x, theta_y
     and theta_z order.
  v: numpy-array
     Array of lateral velocities (shape = (n, 6)), in same order as above
  titles: numpy-array (default = None)
     Array of plot titles (shape = (8, )), in longitudinal and lateral order (see
     description of plotLong and plotLat for details). If None, defaults are used.
  xlabels: numpy-array (default = None)
     Array of plot x-axis labels (shape = (8, )), in same order as for title. If
     None, default labels are used.
  ylabels: numpy-array (default = None)
     Array of plot y-axis labels (shape = (8, )), in same order as for title. If
     None, default labels are used.
  lineType: string (default = '-')
     String containing the line dash type ('-', '.', '--', '-.')
  legends: numpy-array (default = None)
     Array of line labels (shape = (12, )) for longitudinal and lateral quantities
     (see descriptions of plotLong and plotLat). If None, default labels are used.
  axes: numpy-array (default = None)
     Array of matplotlib.pyplot axes (shape = (8, )) to be used for plotting. First
     four are for the longitudinal plots, second four for the lateral plots. If None,
     new axes are generated.
  
  Returns
  -------
  longAxes: numpy-array
     Array containing the axes objects for the longitudinal plots.
  latAxes: numpy-array
     Array containing the axes objects for the lateral plots.
  """
  if axes is not None:
    longAxes = axes[0:4]
    latAxes = axes[4:]
  else:
    longAxes = None
    latAxes = None
  
  if titles is not None:
    longTitles = titles[0:4]
    latTitles = titles[4:]
  else:
    longTitles = None
    latTitles = None
  
  if xlabels is not None:
    longXLabels = xlabels[0:4]
    latXLabels = xlabels[4:]
  else:
    longXLabels = None
    latXLabels = None
  
  if ylabels is not None:
    longYLabels = ylabels[0:4]
    latYLabels = ylabels[4:]
  else:
    longYLabels = None
    latYLabels = None

  if legends is not None:
    longLegends = legends[0:6]
    latLegends = legends[6:]
  else:
    longLegends = None
    latLegends = None
  
  fig1, ax11, ax12, ax13, ax14 = plotLong(t, x[:, [0, 2, 4]], v[:, [0, 2, 4]],
                                          longTitles, longXLabels, longYLabels, lineType,
                                          longLegends, longAxes)
  longAxes = np.array([ax11, ax12, ax13, ax14], dtype=object)

  fig2, ax21, ax22, ax23, ax24 = plotLat(t, x[:, [1, 3, 5]], v[:, [1, 3, 5]],
                                         latTitles, latXLabels, latYLabels, lineType,
                                         latLegends, latAxes)
  latAxes = np.array([ax21, ax22, ax23, ax24], dtype=object)

  return longAxes, latAxes

def plotForces(t: np.ndarray, F: np.ndarray, titles: np.ndarray = None,
                xlabels: np.ndarray = None, ylabels: np.ndarray = None,
                lineType: str = '-', legends: np.ndarray = None, axes: np.ndarray = None):
  """Plot lateral postion and velocity (translations and rotations).

  Parameters
  ----------
  t: numpy-array
     Array of time values (shape = (n, ))
  F: numpy-array
     Array of force values (shape = (n, 6)), in force-x,-y,-z and
     moment-x,-y,-z order.
  titles: numpy-array (default = None)
     Array of plot titles (shape = (2, )), in force, moment order. If None,
     default titles are used.
  xlabels: numpy-array (default = None)
     Array of plot x-axis labels (shape = (2, )), in same order as for titles. If
     None, default labels are used.
  ylabels: numpy-array (default = None)
     Array of plot y-axis labels (shape = (2, )), in same order as for titles. If
     None, default labels are used.
  lineType: string (default = '-')
     String containing the line dash type ('-', '.', '--', '-.')
  legends: numpy-array (default = None)
     Array of line labels (shape = (6, )), in the same order as for F. If None,
     default labels are used.
  axes: numpy-array (default = None)
     Array of matplotlib.pyplot axes (shape = (2, )) to be used for plotting. If None,
     new axes are generated.
  
  Returns
  -------
  fig: matplotlib.pyplot figure
  ax1, ax2: matplotlib.pyplot axes (for plots in order listed for titles)
  """
  if axes is None:
    fig, (ax1, ax2) = plt.subplots(1, 2)
  else:
    ax1, ax2 = tuple(axes)
    fig = ax1.get_figure()

  if titles is None:
    titles = np.array([r'', r''])
  if xlabels is None:
    xlabels = np.array([r'Time, (s)', r'Time, (s)'])
  
  if ylabels is None:
    ylabels = np.array([r'Force Perturbation, (N)', r'Moment Perturbation, (Nm)'])
  
  if legends is None:
    legends = np.array([r'$\delta F_x$', r'$\delta F_y$', r'$\delta F_z$', r'$\delta M_x$', r'$\delta M_y$', r'$\delta M_z$'])
  
  ax1.plot(t, F[:, 0], lineType + 'b', label=legends[0])
  ax1.plot(t, F[:, 1], lineType + 'g', label=legends[1])
  ax1.plot(t, F[:, 2], lineType + 'm', label=legends[2])
  ax1.set_xlabel(xlabels[0])
  ax1.set_ylabel(ylabels[0])
  ax1.set_title(titles[0])
  ax1.legend()

  ax2.plot(t, F[:, 3], lineType + 'b', label=legends[3])
  ax2.plot(t, F[:, 4], lineType + 'g', label=legends[4])
  ax2.plot(t, F[:, 5], lineType + 'm', label=legends[5])
  ax2.set_xlabel(xlabels[1])
  ax2.set_ylabel(ylabels[1])
  ax2.set_title(titles[1])
  ax2.legend()

  return fig, ax1, ax2

def plotForcesSeparate(t: np.ndarray, F: np.ndarray, titles: np.ndarray = None,
                xlabels: np.ndarray = None, ylabels: np.ndarray = None, color: str = 'b',
                lineType: str = '-', legend: np.ndarray = None, axes: np.ndarray = None):
  """Plot lateral postion and velocity (translations and rotations).

  Parameters
  ----------
  t: numpy-array
     Array of time values (shape = (n, ))
  F: numpy-array
     Array of force values (shape = (n, 6)), in force-x,-y,-z and
     moment-x,-y,-z order.
  titles: numpy-array (default = None)
     Array of plot titles (shape = (6, )), in force-x,-y,-z, moment-x,-y,-z order. If None,
     default titles are used.
  xlabels: numpy-array (default = None)
     Array of plot x-axis labels (shape = (6, )), in same order as for titles. If
     None, default labels are used.
  ylabels: numpy-array (default = None)
     Array of plot y-axis labels (shape = (6, )), in same order as for titles. If
     None, default labels are used.
  color: string (default = 'b')
     String containing the line color (valid entries 'b', 'g', 'r', etc., same as plt.plot)
  lineType: string (default = '-')
     String containing the line dash type ('-', '.', '--', '-.')
  legend: string (default = None)
     String containing the legend entry for each plot (which are identical).
  axes: numpy-array (default = None)
     Array of matplotlib.pyplot axes (shape = (6, )) to be used for plotting. If None,
     new axes are generated.
  
  Returns
  -------
  ax1, ax2, ax3, ax4, ax5, ax6: matplotlib.pyplot axes (for plots in order listed for titles)
  """
  if axes is None:
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots()
  else:
    ax1, ax2, ax3, ax4, ax5, ax6 = tuple(axes)

  if titles is None:
    titles = np.array([r'Force, $x-$Direction', r'Force, $y-$Direction', r'Force, $z-$Direction',
                       r'Moment, $x-$Direction', r'Moment, $y-$Direction', r'Moment, $z-$Direction'])
  if xlabels is None:
    xlabels = np.array([r'Time, (s)', r'Time, (s)', r'Time, (s)', r'Time, (s)', r'Time, (s)', r'Time, (s)'])
  
  if ylabels is None:
    ylabels = np.array([r'Force, (N)', r'Force, (N)', r'Force, (N)',
                        r'Moment, (Nm)', r'Moment, (Nm)', r'Moment, (Nm)'])
  
  if legend is None:
    legend = ''
  
  ax1.plot(t, F[:, 0], lineType + color, label=legend)
  ax1.set_xlabel(xlabels[0])
  ax1.set_ylabel(ylabels[0])
  ax1.set_title(titles[0])
  ax1.legend()

  ax2.plot(t, F[:, 1], lineType + color, label=legend)
  ax2.set_xlabel(xlabels[1])
  ax2.set_ylabel(ylabels[1])
  ax2.set_title(titles[1])
  ax2.legend()

  ax3.plot(t, F[:, 2], lineType + color, label=legend)
  ax3.set_xlabel(xlabels[2])
  ax3.set_ylabel(ylabels[2])
  ax3.set_title(titles[2])
  ax3.legend()

  ax4.plot(t, F[:, 3], lineType + color, label=legend)
  ax4.set_xlabel(xlabels[3])
  ax4.set_ylabel(ylabels[3])
  ax4.set_title(titles[3])
  ax4.legend()

  ax5.plot(t, F[:, 4], lineType + color, label=legend)
  ax5.set_xlabel(xlabels[4])
  ax5.set_ylabel(ylabels[4])
  ax5.set_title(titles[4])
  ax5.legend()

  ax6.plot(t, F[:, 5], lineType + color, label=legend)
  ax6.set_xlabel(xlabels[5])
  ax6.set_ylabel(ylabels[5])
  ax6.set_title(titles[5])
  ax6.legend()

  return ax1, ax2, ax3, ax4, ax5, ax6

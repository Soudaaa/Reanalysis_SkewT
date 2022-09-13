from contextlib import ExitStack

from matplotlib.axes import Axes
import matplotlib.transforms as transforms
import matplotlib.axis as maxis
import matplotlib.spines as mspines
from matplotlib.projections import register_projection
import numpy as np


class SkewTTransform(transforms.Affine2D):
    """Perform Skew transform for Skew-T plotting.

    This works in pixel space, so is designed to be applied after the normal plotting
    transformations.
    """

    def __init__(self, bbox, rot):
        """Initialize skew transform.

        This needs a reference to the parent bounding box to do the appropriate math and
        to register it as a child so that the transform is invalidated and regenerated if
        the bounding box changes.
        """
        super().__init__()
        self._bbox = bbox
        self.set_children(bbox)
        self.invalidate()

        # We're not trying to support changing the rotation, so go ahead and convert to
        # the right factor for skewing here and just save that.
        self._rot_factor = np.tan(np.deg2rad(rot))

    def get_matrix(self):
        """Return transformation matrix."""
        if self._invalid:
            # The following matrix is equivalent to the following:
            # x0, y0 = self._bbox.xmin, self._bbox.ymin
            # self.translate(-x0, -y0).skew_deg(self._rot, 0).translate(x0, y0)
            # Setting it this way is just more efficient.
            self._mtx = np.array([[1.0, self._rot_factor, -self._rot_factor * self._bbox.ymin],
                                  [0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]])

            # Need to clear both the invalid flag *and* reset the inverse, which is cached
            # by the parent class.
            self._invalid = 0
            self._inverted = None
        return self._mtx


class SkewXTick(maxis.XTick):
    r"""Make x-axis ticks for Skew-T plots.

    This class adds to the standard :class:`matplotlib.axis.XTick` dynamic checking
    for whether a top or bottom tick is actually within the data limits at that part
    and draw as appropriate. It also performs similar checking for gridlines.
    """

    # Taken from matplotlib's SkewT example to update for matplotlib 3.1's changes to
    # state management for ticks. See matplotlib/matplotlib#10088
    def draw(self, renderer):
        """Draw the tick."""
        # When adding the callbacks with `stack.callback`, we fetch the current
        # visibility state of the artist with `get_visible`; the ExitStack will
        # restore these states (`set_visible`) at the end of the block (after
        # the draw).
        with ExitStack() as stack:
            for artist in [self.gridline, self.tick1line, self.tick2line,
                           self.label1, self.label2]:
                stack.callback(artist.set_visible, artist.get_visible())

            self.tick1line.set_visible(self.tick1line.get_visible() and self.lower_in_bounds)
            self.label1.set_visible(self.label1.get_visible() and self.lower_in_bounds)
            self.tick2line.set_visible(self.tick2line.get_visible() and self.upper_in_bounds)
            self.label2.set_visible(self.label2.get_visible() and self.upper_in_bounds)
            self.gridline.set_visible(self.gridline.get_visible() and self.grid_in_bounds)
            super().draw(renderer)

    @property
    def lower_in_bounds(self):
        """Whether the lower part of the tick is in bounds."""
        return transforms.interval_contains(self.axes.lower_xlim, self.get_loc())

    @property
    def upper_in_bounds(self):
        """Whether the upper part of the tick is in bounds."""
        return transforms.interval_contains(self.axes.upper_xlim, self.get_loc())

    @property
    def grid_in_bounds(self):
        """Whether any of the tick grid line is in bounds."""
        return transforms.interval_contains(self.axes.xaxis.get_view_interval(),
                                            self.get_loc())


class SkewXAxis(maxis.XAxis):
    r"""Make an x-axis that works properly for Skew-T plots.

    This class exists to force the use of our custom :class:`SkewXTick` as well
    as provide a custom value for interval that combines the extents of the
    upper and lower x-limits from the axes.
    """

    def _get_tick(self, major):
        return SkewXTick(self.axes, None, major=major)

    # Needed to properly handle tight bbox
    def _get_tick_bboxes(self, ticks, renderer):
        """Return lists of bboxes for ticks' label1's and label2's."""
        return ([tick.label1.get_window_extent(renderer)
                 for tick in ticks if tick.label1.get_visible() and tick.lower_in_bounds],
                [tick.label2.get_window_extent(renderer)
                 for tick in ticks if tick.label2.get_visible() and tick.upper_in_bounds])

    def get_view_interval(self):
        """Get the view interval."""
        return self.axes.upper_xlim[0], self.axes.lower_xlim[1]


class SkewSpine(mspines.Spine):
    r"""Make an x-axis spine that works properly for Skew-T plots.

    This class exists to use the separate x-limits from the axes to properly
    locate the spine.
    """

    def _adjust_location(self):
        pts = self._path.vertices
        if self.spine_type == 'top':
            pts[:, 0] = self.axes.upper_xlim
        else:
            pts[:, 0] = self.axes.lower_xlim


class SkewXAxes(Axes):
    r"""Make a set of axes for Skew-T plots.

    This class handles registration of the skew-xaxes as a projection as well as setting up
    the appropriate transformations. It also makes sure we use our instances for spines
    and x-axis: :class:`SkewSpine` and :class:`SkewXAxis`. It provides properties to
    facilitate finding the x-limits for the bottom and top of the plot as well.
    """

    # The projection must specify a name.  This will be used be the
    # user to select the projection, i.e. ``subplot(111,
    # projection='skewx')``.
    name = 'skewx'

    def __init__(self, *args, **kwargs):
        r"""Initialize `SkewXAxes`.

        Parameters
        ----------
        args : Arbitrary positional arguments
            Passed to :class:`matplotlib.axes.Axes`

        position: int, optional
            The rotation of the x-axis against the y-axis, in degrees.

        kwargs : Arbitrary keyword arguments
            Passed to :class:`matplotlib.axes.Axes`

        """
        # This needs to be popped and set before moving on
        self.rot = kwargs.pop('rotation', 30)
        super().__init__(*args, **kwargs)

    def _init_axis(self):
        # Taken from Axes and modified to use our modified X-axis
        self.xaxis = SkewXAxis(self)
        self.spines['top'].register_axis(self.xaxis)
        self.spines['bottom'].register_axis(self.xaxis)
        self.yaxis = maxis.YAxis(self)
        self.spines['left'].register_axis(self.yaxis)
        self.spines['right'].register_axis(self.yaxis)

    def _gen_axes_spines(self, locations=None, offset=0.0, units='inches'):
        # pylint: disable=unused-argument
        return {'top': SkewSpine.linear_spine(self, 'top'),
                'bottom': mspines.Spine.linear_spine(self, 'bottom'),
                'left': mspines.Spine.linear_spine(self, 'left'),
                'right': mspines.Spine.linear_spine(self, 'right')}

    def _set_lim_and_transforms(self):
        """Set limits and transforms.

        This is called once when the plot is created to set up all the
        transforms for the data, text and grids.

        """
        # Get the standard transform setup from the Axes base class
        super()._set_lim_and_transforms()

        # This transformation handles the skewing
        skew_trans = SkewTTransform(self.bbox, self.rot)

        # Create the full transform from Data to Pixels
        self.transData += skew_trans

        # Blended transforms like this need to have the skewing applied using
        # both axes, in axes coords like before.
        self._xaxis_transform += skew_trans

    @property
    def lower_xlim(self):
        """Get the data limits for the x-axis along the bottom of the axes."""
        return self.axes.viewLim.intervalx

    @property
    def upper_xlim(self):
        """Get the data limits for the x-axis along the top of the axes."""
        return self.transData.inverted().transform([[self.bbox.xmin, self.bbox.ymax],
                                                    self.bbox.max])[:, 0]


# Now register the projection with matplotlib so the user can select it.
register_projection(SkewXAxes)
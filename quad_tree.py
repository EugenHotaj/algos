"""A QuadTree implementation in python."""

import collections

from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np

Point = collections.namedtuple('Point', 'x y')

class Rectangle(collections.namedtuple('Rectangle', 'x y w h')):

    def contains(self, point):
        """Checks if the given point is contained within the rectangle."""
        return (self.x <= point.x and self.x + self.w >= point.x and 
                self.y <= point.y and self.y + self.h >= point.y)

    def intersects(self, other):
        """Checks if the other rectangle intersects this rectangle."""
        return (self.x < other.x + other.w and
                self.x + self.w > other.x and
                self.y < other.y + other.h and
                self.y + self.h > other.y)

class QuadTree(object):
    def __init__(self, boundary, size=5):
        self._boundary = boundary
        self._size = size
        self._points = []
        self._children = []

    def insert(self, point):
        """Insert the point into the QuadTree in O(logN) time. 
        
        If the tree is at capacity it is split up into 4 subtrees, one for each
        quadrant.

        Args:
            point: The point to add to the Tree.
        Returns:
            Whether the point was successfully added to the tree or children.
        """
        if not self._boundary.contains(point):
            return False
        if len(self._points) < self._size:
            self._points.append(point)
            return True
        # If at max capacity, assign point to children.
        if not self._children:
            x = self._boundary.x
            y = self._boundary.y
            width = self._boundary.w / 2
            height = self._boundary.h / 2
            tl = Rectangle(x, y,  width, height)
            tr = Rectangle(x + width, y, width, height)
            bl = Rectangle(x, y + height, width, height)
            br = Rectangle(x + width, y + height, width, height)
            self._children.append(QuadTree(tl, size=self._size))
            self._children.append(QuadTree(tr, size=self._size))
            self._children.append(QuadTree(bl, size=self._size))
            self._children.append(QuadTree(br, size=self._size))
        for child in self._children:
            # Only insert the point in one child if it falls on a boundary 
            # between two points.
            if child.insert(point):
                return True

    def query(self, boundary):
        """Returns the points contained in the boundary in O(logN) time."""
        points = []
        if self._boundary.intersects(boundary):
            for point in self._points:
                if boundary.contains(point):
                    points.append(point)
            for child in self._children:
                points.extend(child.query(boundary))
        return points

if __name__ == '__main__':
    # Generate points in (-100, 100) for both x and y.
    points = np.random.random(size=(1000, 2)) * 200 - 100

    quad_tree = QuadTree(Rectangle(-100, -100, 200, 200))
    for point in points:
        p = Point(point[0], point[1])
        quad_tree.insert(p)
    query = Rectangle(-50, -50, 100, 100)
    query_points = quad_tree.query(query)

    _, ax = plt.subplots(1)
    # Color query points red and non-query points blue.
    colors = []
    for point in points:
        p = Point(point[0], point[1])
        if p in query_points:
            colors.append('red')
        else:
            colors.append('blue')
    plt.scatter(points[:, 0], points[:, 1], s=1, c=colors)
    # Draw the query rectangle.
    rect = patches.Rectangle((query.x, query.y), query.w, query.h, linewidth=1, 
                             edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

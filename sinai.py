import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation

from itertools import combinations

import matplotlib
matplotlib.use('tkagg')


class Vector():
    """2D vector
    """
    def __init__(self, x, y):
        self.r = np.array((x, y))

    def get_distance(self, other):
        # return np.hypot(*(self.r - other.r))
        return np.linalg.norm(self.r - other.r)

    def __add__(self, other):
        return self.r + other

    def __radd__(self, other):
        return other + self.r

    def __mul__(self, val):
        return self.r * val

    def __rmul__(self, val):
        return val * self.r

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

    @property
    def x(self):
        return self.r[0]

    @x.setter
    def x(self, value):
        self.r[0] = value

    @property
    def y(self):
        return self.r[1]

    @y.setter
    def y(self, value):
        self.r[1] = value


class Particle:
    """A 2D circle."""

    def __init__(self, x, y, vx, vy,
                 mass=None, radius=0.01,
                 styles={'facecolor': 'b', 'edgecolor': 'b', 'fill': False}):
        """
            x, y:
                coordinates
            vx, vy:
                velocity along x and y
            radius:
                radius of the circles
            styles: [dict]
                argument to matplotlib.patches.Circle
        """

        self.r = Vector(x, y)
        self.v = Vector(vx, vy)
        self.radius = radius
        if not mass:
            self.mass = self.radius**2
        else:
            self.mass = mass
        self.styles = styles

    def __str__(self):
        return f"Particle(r={self.r}, v={self.v}, "\
               f"m={self.mass}, r={self.radius})"

    def overlaps(self, other):
        """
            Check if current particle overlaps with another particle
        """

        return Vector.get_distance(self.r,
                                   other.r) <= self.radius + other.radius

    def draw(self, ax):
        """Add this Particle's Circle patch to the Matplotlib Axes ax."""

        circle = Circle(xy=self.r.r, radius=self.radius, **self.styles)
        ax.add_patch(circle)
        return circle

    def forward(self, dt):
        """Advance the Particle's position forward in time by dt."""
        self.r.r = self.r + (self.v * dt)


class Simulation():

    ParticleClass = Particle

    def __init__(self,
                 n=1, radius=0.01, speed_factor=1,
                 width=1, height=1,
                 styles=None):
        self.dt = 0.01

        self.WIDTH = width
        self.HEIGHT = height

        self.initialize_particles(n, radius,
                                  speed_factor,
                                  styles)

    def place_particle(self, r, x=None, y=None,
                       m=0, speed_factor=1,
                       styles=None):
        if not x and not y:
            x, y = r + (1 - 2 * r) * np.random.random(2)

        vr = 0.1 * np.sqrt(np.random.random()) + 0.05
        vphi = 2 * np.pi * np.random.random()
        vx, vy = vr * np.cos(vphi), vr * np.sin(vphi)
        particle = self.ParticleClass(x, y,
                                      vx * speed_factor, vy * speed_factor,
                                      m, r, styles)

        for p2 in self.particles:
            if p2.overlaps(particle):
                break
        else:
            self.particles.append(particle)
            return True
        return False

    def initialize_particles(self, n, radius,
                             speed_factor=1,
                             styles=None):
        try:
            _ = (r for r in radius)
            assert n == len(radius)
        except TypeError:
            # r isn't iterable: turn it into a generator that returns the
            # same value n times.
            def r_gen(n, radius):
                for i in range(n):
                    yield radius
            radius = r_gen(n, radius)

        self.n = n
        self.particles = []

        radius[0] = self.WIDTH / 4

        for i, r in enumerate(radius):
            # place every particle at a random location
            # except for the very first one
            if i == 0:
                s = 0
                x0 = self.WIDTH / 2
                y0 = self.HEIGHT / 2
                m = 10000
            else:
                s = speed_factor
                x0 = 0
                y0 = 0
                m = 0
            while not self.place_particle(r, x=x0, y=y0,
                                          m=m, speed_factor=s,
                                          styles=styles):
                pass

    def boundary_collisions(self, p):
        """Bounce the particles off the walls.
        We assume that the collision is elastic."""
        if p.r.x - p.radius < 0:
            p.r = Vector(p.radius, p.r.y)
            p.v = Vector(-p.v.x, p.v.y)
        if p.r.x + p.radius > 1:
            p.r = Vector(1 - p.radius, p.r.y)
            p.v = Vector(-p.v.x, p.v.y)
        if p.r.y - p.radius < 0:
            p.r = Vector(p.r.x, p.radius)
            p.v = Vector(p.v.x, -p.v.y)
        if p.r.y + p.radius > 1:
            p.r = Vector(p.r.x, 1 - p.radius)
            p.v = Vector(p.v.x, -p.v.y)

    def change_velocities(self, p1, p2):
        """
        Particles p1 and p2 collided elastically: update their
        velocities.
        """
        m1, m2 = p1.mass, p2.mass
        M = m1 + m2
        r1, r2 = p1.r.r, p2.r.r
        d = np.linalg.norm(r1 - r2)**2
        v1, v2 = p1.v.r, p2.v.r
        u1 = v1 - 2 * m2 / M * np.dot(v1 - v2, r1 - r2) / d * (r1 - r2)
        u2 = v2 - 2 * m1 / M * np.dot(v2 - v1, r2 - r1) / d * (r2 - r1)
        p1.v = Vector(u1[0], u1[1])
        p2.v = Vector(u2[0], u2[1])

    def particle_collisions(self):
        """Bounce the particle against another particle.
        Again, the collision is elastic.
        """
        pairs = combinations(range(self.n), 2)
        for i, j in pairs:
            if self.particles[i].overlaps(self.particles[j]):
                self.change_velocities(self.particles[i], self.particles[j])

    def apply_forces(self):
        """Override this method to accelerate the particles."""
        pass

    def init(self):
        """Initialize the animation."""

        self.circles = []
        for particle in self.particles:
            self.circles.append(particle.draw(self.ax))
        return self.circles

    def setup(self):
        self.fig, self.ax = plt.subplots()
        for s in ['top', 'bottom', 'left', 'right']:
            self.ax.spines[s].set_linewidth(2)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(0, self.WIDTH)
        self.ax.set_ylim(0, self.HEIGHT)
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])

    def animate(self, i):
        for i, p in enumerate(self.particles):
            p.forward(self.dt)
            self.boundary_collisions(p)
            self.circles[i].center = p.r.r
        self.particle_collisions()
        self.apply_forces()

        return self.circles

    def run(self, save=False, interval=1, filename='sinai.mp4'):
        self.setup()
        anim = animation.FuncAnimation(self.fig, self.animate,
                                       init_func=self.init, frames=800,
                                       interval=interval, blit=True)

        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, bitrate=1800)
            anim.save(filename, writer=writer)
        else:
            plt.show()


if __name__ == '__main__':
    n = 10
    radii = np.random.random(n) * 0.03 + 0.02

    styles = {'color': 'C9',
              'linewidth': 2,
              'fill': 0}

    s = Simulation(n=n, radius=radii, speed_factor=1, styles=styles)
    s.run()

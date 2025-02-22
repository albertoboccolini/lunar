import pygame
import math
import random
from game.constants import WINDOW_WIDTH, WINDOW_HEIGHT, CEILING, LANDING_PAD_TOP, ROTATION_SPEED, THRUST_POWER, \
    GRAVITY


def make_lander_rounded(surface, color, points, radius, segments=10):
    """
    Draw a triangle with rounded corners inside.

    - surface: the surface to draw on.
    - color: color (RGB) of the triangle.
    - points: list of 3 tuples (x, y) of the vertices of the triangle.
    - radius: radius of the rounding.
    - segments: number of segments to approximate each arc.
    """

    def get_offset_point(p_from, p_to, r):
        # Calculate a point along the side from p_from to p_to at distance r from p_from
        dx = p_to[0] - p_from[0]
        dy = p_to[1] - p_from[1]
        d = math.hypot(dx, dy)
        if d == 0:
            return p_from
        return p_from[0] + (dx / d) * r, p_from[1] + (dy / d) * r

    final_points = []
    n = len(points)

    for i in range(n):
        # Get the current vertex and its adjacent ones
        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[(i + 1) % n]

        # Calculate the offset points along the sides (to move the corner inside)
        offset1 = get_offset_point(p_curr, p_prev, radius)
        offset2 = get_offset_point(p_curr, p_next, radius)

        # Calculate the angles of the two vectors starting from the vertex towards the offsets
        angle1 = math.atan2(offset1[1] - p_curr[1], offset1[0] - p_curr[0])
        angle2 = math.atan2(offset2[1] - p_curr[1], offset2[0] - p_curr[0])

        # Ensures a positive (inner) arc; adjust the interval if necessary
        if angle2 < angle1:
            angle2 += 2 * math.pi

        # Generate points for the arc that replaces the vertex
        arc_points = []
        for j in range(segments + 1):
            t = j / segments
            angle = angle1 + t * (angle2 - angle1)
            arc_x = p_curr[0] + radius * math.cos(angle)
            arc_y = p_curr[1] + radius * math.sin(angle)
            arc_points.append((arc_x, arc_y))

        # Add the first offset point and then the arc points
        final_points.append(offset1)
        final_points.extend(arc_points)
        # The point offset2 will be added as the starting point of the next vertex arc

    pygame.draw.polygon(surface, color, final_points)


def draw_lander(lander_img):
    lander_points = [(20, 0), (0, 60), (40, 60)]
    make_lander_rounded(lander_img, (15, 23, 42), lander_points, 5)


def create_stars(stars):
    num_stars = 150
    for _ in range(num_stars):
        star = {
            'x': random.randint(0, WINDOW_WIDTH),
            'y': random.randint(0, WINDOW_HEIGHT),
            'radius': random.randint(1, 3),
            'speed': random.uniform(10, 20)
        }
        stars.append(star)


def draw_landing_pad(screen):
    landing_pad = pygame.Rect(WINDOW_WIDTH / 2 - 200, LANDING_PAD_TOP, 400, 10)
    pygame.draw.rect(screen, (15, 23, 42), landing_pad, border_radius=10)


def verify_lander_crash(self):
    # Ceiling
    if self.pos.y - self.height / 2 <= CEILING:
        self.pos.y = CEILING + self.height / 2
        self.done = True
        self.fitness -= 1000

    if self.pos.y + self.height / 2 >= LANDING_PAD_TOP:
        self.pos.y = LANDING_PAD_TOP - self.height / 2
        self.done = True

        # Low vertical speed (< 100 pixels/s) and almost vertical angle (±15°)
        if abs(self.vel.y) < 100 and (abs(self.angle % 360) < 15 or abs((self.angle % 360) - 360) < 15):
            self.fitness += 1000
        else:
            self.fitness -= 1000


class LunarLanderEnv:
    def __init__(self):
        self.fitness = 0
        self.done = False
        self.angle = None
        self.vel = None
        self.pos = None
        self.reset()

        self.lander_img = pygame.Surface((40, 60), pygame.SRCALPHA)
        draw_lander(self.lander_img)

        self.width = 40
        self.height = 60

        self.stars = []
        create_stars(self.stars)

    def update_stars(self, dt):
        """Updates the position of the stars, sliding them from right to left."""
        for star in self.stars:
            star['x'] -= star['speed'] * dt
            if star['x'] < 0:
                star['x'] = WINDOW_WIDTH
                star['y'] = random.randint(0, WINDOW_HEIGHT)

    def reset(self):
        self.pos = pygame.Vector2(WINDOW_WIDTH / 2, 100)
        self.vel = pygame.Vector2(0, 0)
        self.angle = random.uniform(0, 360)
        self.done = False
        self.fitness = 0

    def step(self, action, dt):
        """
        action: tuple (rotate_left, rotate_right, thrust)
        dt: delta time in secondi
        """

        # Rotation
        if action[0]:
            self.angle += ROTATION_SPEED * dt
        if action[1]:
            self.angle -= ROTATION_SPEED * dt

        # Thrust application
        if action[2]:
            rad = math.radians(self.angle)
            thrust_vec = pygame.Vector2(-math.sin(rad), -math.cos(rad))
            self.vel += thrust_vec * THRUST_POWER * dt

        # Gravity application
        self.vel.y += GRAVITY * dt

        # Update position
        self.pos += self.vel * dt

        # Horizontal limits
        if self.pos.x < 0:
            self.pos.x = 0
            self.vel.x = 0
        elif self.pos.x > WINDOW_WIDTH:
            self.pos.x = WINDOW_WIDTH
            self.vel.x = 0

        verify_lander_crash(self)

        # Increases survival fitness (encouraging more time in flight)
        self.fitness += dt

    def get_observation(self):
        """
        Returns the normalized state:
        - Position x and y
        - Velocity x and y
        - sin(angle) and cos(angle) to represent the orientation
        - Altitude (distance from the platform)
        """
        altitude = LANDING_PAD_TOP - (self.pos.y + self.height / 2)
        return [
            self.pos.x / WINDOW_WIDTH,
            self.pos.y / WINDOW_HEIGHT,
            self.vel.x / 300,  # Normalized speed
            self.vel.y / 300,
            math.sin(math.radians(self.angle)),
            math.cos(math.radians(self.angle)),
            altitude / 100
        ]

    def render_game(self, screen, dt):
        """Draw the game environment."""

        self.update_stars(dt)
        screen.fill((255, 255, 255))
        star_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        for star in self.stars:
            pygame.draw.circle(star_surface, (0, 0, 0, 50), (int(star['x']), int(star['y'])), star['radius'])
        screen.blit(star_surface, (0, 0))

        # Ceiling
        pygame.draw.line(screen, (255, 255, 255), (0, CEILING), (WINDOW_WIDTH, CEILING), 2)

        draw_landing_pad(screen)

        # Lander
        rotated_img = pygame.transform.rotozoom(self.lander_img, self.angle, 1)
        rect = rotated_img.get_rect(center=(self.pos.x, self.pos.y))
        screen.blit(rotated_img, rect.topleft)
        pygame.display.flip()

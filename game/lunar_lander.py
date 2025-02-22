import pygame
import math
import random
from game.constants import WINDOW_WIDTH, WINDOW_HEIGHT, CEILING, LANDING_PAD_TOP, ROTATION_SPEED, THRUST_POWER, \
    GRAVITY, MAX_SPEED_TO_LAND, MAX_ANGLE_TO_LAND, SLOW_LANDING_BONUS, SLOW_LANDING_PENALTY, RANDOMIZE_LANDER_ANGLE


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

    # Landing pad
    if self.pos.y + self.height / 2 >= LANDING_PAD_TOP:
        self.pos.y = LANDING_PAD_TOP - self.height / 2
        self.done = True
        landing_bonus = SLOW_LANDING_PENALTY

        if abs(self.vel.y) < 100:
            landing_bonus = SLOW_LANDING_BONUS

        # Only if vertical speed less than 100 and vertical angle near 15 degrees
        safe_landing = abs(self.vel.y) < MAX_SPEED_TO_LAND and (
                abs(self.angle % 360) < MAX_ANGLE_TO_LAND or abs((self.angle % 360) - 360) < MAX_ANGLE_TO_LAND)

        if safe_landing:
            self.fitness += 1000 + landing_bonus
            if not hasattr(self, "win_count"):
                self.win_count = 0
            self.win_count += 1

        if not safe_landing:
            self.fitness -= 1000 + landing_bonus


class LunarLanderEnv:
    def __init__(self):
        self.fitness = 0
        self.done = False
        self.angle = None
        self.vel = None
        self.pos = None
        self.reset()

        self.lander_img = pygame.image.load('game/images/lander.png').convert_alpha()

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
        self.angle = random.uniform(0, RANDOMIZE_LANDER_ANGLE)
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

        if self.pos.x > WINDOW_WIDTH:
            self.pos.x = WINDOW_WIDTH
            self.vel.x = 0

        verify_lander_crash(self)

        # Encouraging speed velocity minor than MAX_SPEED_TO_LAND
        if abs(self.vel.y) < MAX_SPEED_TO_LAND:
            self.fitness += dt

        # Encouraging angle minor than 15
        if abs(self.angle % 360) < MAX_ANGLE_TO_LAND or abs((self.angle % 360) - 360) < MAX_ANGLE_TO_LAND:
            self.fitness += dt

        if self.fitness < 0:
            self.fitness = 0

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

        # Stats
        font = pygame.font.SysFont("Arial", 16)
        angle_text = font.render(f"Angle: {self.angle:.2f}", True, (0, 0, 0))
        speed_text = font.render(f"Speed: {self.vel.length():.2f}", True, (0, 0, 0))
        fitness_text = font.render(f"Fitness: {self.fitness:.2f}", True, (0, 0, 0))
        screen.blit(angle_text, (10, 10))
        screen.blit(speed_text, (10, 40))
        screen.blit(fitness_text, (10, 70))

        outcome_str = "Dangerous Landing"
        if abs(self.vel.y) < MAX_SPEED_TO_LAND and (
                abs(self.angle % 360) < MAX_ANGLE_TO_LAND or abs((self.angle % 360) - 360) < MAX_ANGLE_TO_LAND):
            outcome_str = "Safe Landing"

        outcome_text = font.render(f"Status: {outcome_str}", True, (0, 0, 0))
        screen.blit(outcome_text, (10, 100))

        pygame.display.flip()

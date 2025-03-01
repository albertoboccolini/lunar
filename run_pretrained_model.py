import pygame
import neat
import pickle
from game.constants import WINDOW_WIDTH, WINDOW_HEIGHT
from game.lunar_lander import LunarLanderEnv
from utils import LUNAR_MODEL_PATH, start_game_observation


def run_winner(config_file, winner_path=LUNAR_MODEL_PATH):
    """
    Runs the Lunar Lander game simulation using the winning neural network model.
    :param config_file: The file where the NEAT config is located.
    :param winner_path: The file where the winning neural network model is located.
    """
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Load the winning genome from the pickle file
    with open(winner_path, "rb") as f:
        winner = pickle.load(f)

    neural_network = neat.nn.FeedForwardNetwork.create(winner, config)

    pygame.init()
    pygame.display.set_icon(pygame.image.load('./game/images/icon.png'))
    pygame.display.set_caption('Lunar')
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    game = LunarLanderEnv()

    waiting_for_reset = False
    wait_start_time = 0

    while True:
        delta_time = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if not waiting_for_reset:
            start_game_observation(game, neural_network, delta_time)
            if game.done:
                waiting_for_reset = True
                wait_start_time = pygame.time.get_ticks()

        if waiting_for_reset and pygame.time.get_ticks() - wait_start_time >= 2000:
            game.reset()
            waiting_for_reset = False

        game.render_game(screen, delta_time)


if __name__ == "__main__":
    run_winner("config-feedforward.txt")

import os
import pygame
import neat
import pickle
from game.constants import WINDOW_WIDTH, WINDOW_HEIGHT, SIMULATION_MAX_STEPS
from game.lunar_lander import LunarLanderEnv
from utils import SHOULD_RENDER_SIMULATION, WIN_THRESHOLD, LUNAR_MODEL_PATH, start_game_observation


def eval_genomes(genomes, config):
    """
    Evaluation function for NEAT. For each genome, a game is simulated
    with real-time rendering.
    """
    pygame.init()
    pygame.display.set_icon(pygame.image.load('./game/images/icon.png'))
    pygame.display.set_caption('Lunar')
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    for genome_id, genome in genomes:
        neural_network = neat.nn.FeedForwardNetwork.create(genome, config)
        game = LunarLanderEnv()
        steps = 0

        while not game.done and steps < SIMULATION_MAX_STEPS and getattr(game, 'win_count', 0) < WIN_THRESHOLD:
            while not game.done and steps < SIMULATION_MAX_STEPS:
                delta_time = 1 / 60.0

                if SHOULD_RENDER_SIMULATION:
                    delta_time = clock.tick(60) / 1000.0
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            quit()

                start_game_observation(game, neural_network, delta_time)

                if SHOULD_RENDER_SIMULATION:
                    game.render_game(screen, delta_time)
                steps += 1

            if getattr(game, 'win_count', 0) < WIN_THRESHOLD:
                game.reset()

        genome.fitness = game.fitness
        genome.win_count = getattr(game, 'win_count', 0)
        if genome.win_count > 0:
            print(f"Genome {genome_id} has obtained {genome.win_count} wins.")
    pygame.quit()


def run_neat(config_file):
    """
    Load the configuration, create the NEAT population and start training.
    """
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    winner = pop.run(eval_genomes, 100)
    if hasattr(winner, 'win_count') and winner.win_count >= WIN_THRESHOLD:
        os.makedirs(os.path.dirname(LUNAR_MODEL_PATH), exist_ok=True)
        with open(LUNAR_MODEL_PATH, "wb") as f:
            pickle.dump(winner, f)
        print("Model saved!", winner)


if __name__ == "__main__":
    run_neat("config-feedforward.txt")

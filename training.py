import os
import pygame
import neat
import pickle
from game.constants import WINDOW_WIDTH, WINDOW_HEIGHT, SIMULATION_MAX_STEPS
from game.lunar_lander import LunarLanderEnv
from utils import SHOULD_RENDER_SIMULATION, start_game_observation, LUNAR_MODEL_PATH, WIN_THRESHOLD


def eval_genomes(genomes, config):
    """
    Continuous evaluation based exclusively on what the game defines.
    - Each genome keeps being evaluated until the lander crashes or reaches 5 consecutive safe landings.
    - Fitness and win count are directly taken from the game.
    :param genomes: List of Genome objects
    :param config: NEAT config object
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
        consecutive_wins = 0

        while steps < SIMULATION_MAX_STEPS:
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

            if game.done:
                if game.fitness <= 0:
                    break  # Immediate termination

                # Safe landing
                consecutive_wins += 1
                if consecutive_wins >= WIN_THRESHOLD:
                    break  # Genome is stable, move to the next

                game.reset()

        genome.fitness = game.fitness
        genome.win_count = getattr(game, 'win_count', 0)

    # After each generation, check if there is a super genome to stop training early
    best_genome = max((genome for _, genome in genomes), key=lambda g: g.fitness, default=None)

    if best_genome and best_genome.fitness >= 1000 and getattr(best_genome, 'win_count', 0) >= WIN_THRESHOLD:
        save_winner(best_genome)
        print(
            f"Training complete: Best genome with fitness {best_genome.fitness:.2f} and {best_genome.win_count} wins "
            f"saved.")
        raise neat.CompleteExtinctionException

    pygame.quit()


def run_neat(config_file):
    """
    Start the NEAT training with continuous evaluation, up to 100 generations.
    :param config_file: The file where the NEAT config is located.
    """
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    try:
        population.run(eval_genomes, 100)
    except neat.CompleteExtinctionException:
        return


def save_winner(genome):
    os.makedirs(os.path.dirname(LUNAR_MODEL_PATH), exist_ok=True)
    with open(LUNAR_MODEL_PATH, "wb") as f:
        pickle.dump(genome, f)


if __name__ == "__main__":
    run_neat("config-feedforward.txt")

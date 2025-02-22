import pygame
import neat
from game.constants import WINDOW_WIDTH, WINDOW_HEIGHT, SIMULATION_MAX_STEPS
from game.lunar_lander import LunarLanderEnv


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
    render = True  # Set to True to view the simulation

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = LunarLanderEnv()
        steps = 0
        while not env.done and steps < SIMULATION_MAX_STEPS:
            # Event management and time update
            if render:
                delta_time = clock.tick(60) / 1000.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
            else:
                delta_time = 1 / 60.0

            # Get state and calculate action from network
            obs = env.get_observation()
            output = net.activate(obs)
            rotate_left = output[0] > 0.5
            rotate_right = output[1] > 0.5
            thrust = output[2] > 0.5
            action = (rotate_left, rotate_right, thrust)
            env.step(action, delta_time)
            if render:
                env.render_game(screen, delta_time)
            steps += 1
        genome.fitness = env.fitness
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
    winner = pop.run(eval_genomes, 50)
    print("Winner:", winner)


if __name__ == "__main__":
    run_neat("config-feedforward.txt")

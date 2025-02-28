WIN_THRESHOLD = 10
LUNAR_MODEL_PATH = "./models/lunar.pkl"
SHOULD_RENDER_SIMULATION = False  # Set to True to see game during the training


def start_game_observation(game, neural_network, delta_time):
    obs = game.get_observation()
    output = neural_network.activate(obs)
    rotate_left = output[0] > 0.2
    rotate_right = output[1] > 0.2
    thrust = output[2] > 0.2
    action = (rotate_left, rotate_right, thrust)
    game.step(action, delta_time)

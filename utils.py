WIN_THRESHOLD = 5
LUNAR_MODEL_PATH = "./models/lunar.pkl"
SHOULD_RENDER_SIMULATION = True  # Change this to False for faster training


def start_game_observation(game, neural_network, delta_time):
    obs = game.get_observation()
    output = neural_network.activate(obs)
    rotate_left = output[0] > 0.5
    rotate_right = output[1] > 0.5
    thrust = output[2] > 0.5
    action = (rotate_left, rotate_right, thrust)
    game.step(action, delta_time)

<div  align="center">
   <a href="https://albertoboccolini.com" target="_blank">
      <img alt="lunar" height="150" src="game/images/icon.png">
   </a>
</div>

# Lunar

A collection of scripts to train a NEAT neural network to play *Lunar Landing*, a game where the player
controls a spaceship during landing. To win, the spaceship must adhere to specific landing rules.

## Train the network

1. **Install [pipenv](https://pipenv.pypa.io/en/latest/installation.html)**:
   Pipenv is used to handle the virtual environment and the necessary packages.

2. **Install necessary packages from `Pipfile`**

   ```bash
   pipenv install
   ```

3. **Run the training script:** on complete the model will be saved in `./models/lunar.pkl`

   ```bash
   pipenv run python training.py
   ```

## Use the pretrained model

1. **Install [pipenv](https://pipenv.pypa.io/en/latest/installation.html)** (skip if already done):
   Pipenv is used to handle the virtual environment and the necessary packages.

2. **Install necessary packages from `Pipfile`** (skip if already done)

   ```bash
   pipenv install
   ```

3. **Run the demo script** (only if you have already trained the model): It will open the game where the pretrained
   model plays an infinite loop of games

   ```bash
   pipenv run python run_pretrained_model.py
   ```

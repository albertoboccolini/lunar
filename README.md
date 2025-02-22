<a href="https://albertoboccolini.com" target="_blank" style="display: flex; justify-content: center">
  <img width="150" height="150" src="game/images/icon.png" style="text-align: center">
</a>

# Lunar

A collection of scripts to train a NEAT neural network to play *Lunar Landing*, a game where the player
controls a spaceship during landing. To win, the spaceship must adhere to specific landing rules.

## Getting started

1. **Install [pipenv](https://pipenv.pypa.io/en/latest/installation.html)**:
   Pipenv is used to handle the virtual environment and the necessary packages.

2. **Install necessary packages from `Pipfile`**

   ```bash
   pipenv install
   ```

3. **Run the training script:**

   ```bash
   pipenv run python training.py
   ```
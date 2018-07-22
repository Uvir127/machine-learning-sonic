# machine-learning-sonic
As part of a Machine Learning Project we decided to implement different reinforcement learning algorithms on the Sonic the Hedgehog game.

The following packages need to be installed:
  gym-retro
  gym-retro local (from open AI website that is used for the competition)
  pytorch
  torchvision
  OpenCV
  tqdm
  python3

  pip3 install --user torch torchvision gym tqdm opencv-python gym-retro
  git clone --recursive https://github.com/openai/retro-contest.git
  pip3 install -e "retro-contest/support[docker,rest]"

The ROM for the game being used needs to be imported into the gym-retro emunlator before running the agent
  python3 -m retro.import <path to ROM>
  *Note: The ROM needs to be of format .md
        This means it needs to be a Japanese ROM

More Details can be found here :
  https://contest.openai.com/details

To run the program use, for example:
  python3 sarsaSanic.py
  
However, to run the DQN programs run the train files, for example:
  python3 train.py

To view the game being run make sure that there is a "env.render()" line after every "env.step(action)" function.

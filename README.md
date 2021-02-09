# README #

### What is this repository for? ###
This project is intended for learning pruposes. Algorithms are generally written to make
concepts clear, rather than to optimize speed. Writing them helped me solidify my understanding,
and may help others who just need a starting point and working implementation with all the 
guts exposed.

### What this repository is NOT for? ###
This is NOT a realistic toolkit. TensorFlow2 is out there, and it's more efficient, 
parallelized, and feature-rich than I could ever hope to make this. Let's not reinvent
the wheel. 

To that end, there is some clean-up and standardization that *I would
like to get to in theory, but probably never will*. The earliest components I created 
were generally the simplest, and now that I'm more knowledgeable, I can see the how many 
features would need to be added, to standardize the interface with everything I would like.
Adding these would be fun, but I would learn less, and learning was always the point of this
project.

### Future Work ###
There are still TODO's throughout, though most are low priority interface cleanups.
The biggest active priority is cleaning this up so that someone could run it fairly easily.

### Getting Started ###
1) Pull down the project. 
2) If I created a yaml, or environment.bat, go ahead and run that (so far, I haven't though)
    1) In case I don't get around to creating a proper script/env, you'll need:
        1) python (3.8, due to a gym dependency)
        2) numpy
        3) gym: The atari games, and the lunarLander take additional plugins
            1) gym : pip install gym
		        1) Requires Pillow 7.2 (doesn't work with 8.1). Since Pillow 7.2 requires Python <=3.8, 
		        this means you need to revert the build to 3.8 (at least as of Jan 2021)
                2) Changing python versions (especially downgrading) can cause lots of problems...
                You may want to consider creating a new environment and performing fresh installs
                for all dependencies:
                3) gym[Atari] on windows: https://github.com/openai/gym/issues/1218
                4) gym[Box2D] for lunar lander: 
			2) pygame works fine with python 3.8, as long as you install into a python3 env. If you 
			installed on a different version, you'll need to uninstall and re-install after updating
			python.
                1) > conda install swig # needed to build Box2D in the pip install
                2) > pip install box2d-py # a repackaged version of pybox2d
3) The Trainers run the whole thing (a bit like TensorFlow Drivers), but each of the agents
has a __main__ script, which is probably the best entry point to start playing/testing.


### How to use this library ###
If you just want to play around with and learn about some agents, then after following 
the __Getting Started__ section, and start playing with agents:
    1) The EnergyPumpingMountainCarAgent is a good example of a "dumb" agent, just to start
    understanding how the gym environments work. 
    2) You can also do manual control with the HumanAgent. Depending on the environment,
    key mappings are arrow keys, or wasd, and space_bar.
    3) The first Reinforcement Learning agent I recommend starting with is SemiGradientTdControlAgents
    This is not actually the simplest, but it just works on a lot of these simplest problems. Default 
    settings should solve most problems in about 500 iterations or less, though it obviously depends on the problem
    and the hyperparameters.
        1) Initially, just ignore the replay model 
            1) For those used to working with TensorFlow,
        the model in this library is analogous to the replaybuffer in tensorFlow, and the Approximator
        is what TensorFlow and most literature would call the Model. I had named a lot of things before
        realizing this.
    4) After that, you can start exploring other agents. 
        1) The NeuralNets are the most powerful, and generalizable
        2) The MonteCarloQAgent is conceptually simplest (though it's less stable with most problems,
        so it's finicky to actually play with if you're just learning)
    
* Learning RL
* 0.1.0
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### Additional Resources ###
Most of this is based on courses I've taken online. Ping me if you want details, or discussion
about how to learn this stuff in general. The most directly relevant courses for this library
are:
* Series on Coursera: https://www.coursera.org/specializations/reinforcement-learning
* Free Canonical Text: http://incompleteideas.net/book/the-book.html
* Colab: If anything takes too long (your computer is struggling), try out google colab.
    - Most agents on most environments in this project train in 1-30 minutes


### Contribution guidelines ###
* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###
* James Trevor Clark (I go by Trevor)
import typing
import numpy as np
import pandas as pd

from brian2 import *
# run in compiled mode
set_device('cpp_standalone')


class SVG_Model(object):

    MODULES = (
        'distress',
        'social_behavioral',
        'body_ownership',
        'gender',
        'culture',
        'feedback'
    )

    CONNECTIVITY = np.array(
        [
        [0, 1, 1, 2, 0, 0], # distress ->
        [1, 0, 1, 2, 0, 0], # social_behavioral ->
        [1, 1, 0, 2, 0, 0], # body_ownership ->
        [2, 2, 2, 0, 0, 0], # gender
        [1, 1, 0, 2, 0, 0], # culture
        [1, 1, 0, 2, 0, 0]  # feedback
        ],
        dtype=int
    ) # type: np.ndarray
    """
    Connectivity matrix between modules. 
    Axes are indexed by :attr:`SVG_Model.MODULES`.
    Synapses connect from axis 0 (rows) to axis 1 (columns).
    "Minor" connections are denoted with 1, 
    "Major" connections are denoted with 2.
    Weights are constructed from the :attr:`.weights` parameter.
    """


    def __init__(self, size:typing.Union[int, typing.Dict[str, int]]=1,
                 weights:typing.Tuple[float, float] = (0.5, 1.),
                 noise_amplitude: typing.Union[float, typing.Dict[str, float]]=0.1,
                 tau: typing.Union[float, typing.Dict[str, float]] = 1,
                 delay: typing.Union[float, typing.Dict[str, float]] = 1,
                 resting: typing.Union[float, typing.Dict[str, float]] = 1,
                 random_init: bool = True
                 ):
        """
        The model is composed of (nonspiking) neuron groups according the modules described in the paper,
        which are connected with synapses with two fixed weights (the black and gray lines).

        Each neuron in each module is governed by a differential equation that exponentially returns
        it to the voltage set by :attr:`.resting` , with stochastic differential noise of amplitude :attr:`.noise_amplitude`

        Postsynaptic neurons receive the voltage of the pre-synaptic neuron with some :attr:`.delay` scaled by
        :attr:`.weights`

        Args:
            size (int, dict): Size of modules, if an int (default: 1), size for all modules,
                otherwise, a dict with {'module_name': size}

                .. note ::

                    Only modules of size == 1 are implemented currently, this is a placeholder for different connectivity modes

            weights (tuple): Weights for ("minor", "major") connections, tuple of two floats
            noise_amplitude (float, dict): Amplitude of noise, if float, noise for all.
                otherwise a dict with {'module_name': amplitude}
            tau (float, dict): Time constant (s) for decay to :attr:`.resting` value, if float, same for all modules,
                otherwise a dict with {'module_name': tau}
            delay (float, dict): delay (s) of module *output*, if float, same for all modules,
                otherwise a dict with {'module_name': delay}
            resting (float, dict): resting value to decay 2. if float, same for all modules,
                otherwise a dict with {'module_name': resting}
            random_init (bool): Whether neurons should be initialized with some random initial value (default: True)


        Attributes:
            modules (dict): Dictionary mapping module name to :class:`brian2.NeuronGroup` object
            synapses (dict): Dictionary mapping ('from', 'to') module name tuples to their :class:`brian2.Synapse` object
            monitors (dict): Dictionary mapping module name to a brian ``StateMonitor`` that stores the values of a run

        """
        # create private attributes
        self._size = None
        self._noise_amplitude = None
        self._tau = None
        self._delay = None
        self._resting = None

        # assign arguments
        self.size = size
        self.weights = weights
        self.noise_amplitude = noise_amplitude
        self.tau = tau
        self.delay = delay
        self.resting = resting
        self.random_init = random_init # type: bool

        # create public attributes
        self.modules = {} # type: typing.Dict[str, NeuronGroup]
        self.synapses = {} # type: typing.Dict[str, Synapses]
        self.monitors = {} # type: typing.Dict[str, StateMonitor]

        # network used to keep track of objects created in methods
        self.network = Network() # type: typing.Optional[Network]

        # initialize network
        self._init_neurons()

    def run(self, duration: float = 1, dt: float = 0.001, reset: bool = True):
        """
        Run the model for ``duration`` seconds.

        Values are recorded to monitors in :attr:`.monitors

        Args:
            duration (float): Seconds to run model
            dt (float): timestep size of model simulation in seconds
            reset (bool): whether to reset the network to its initial state on start
        """

        # Make monitors for each population
        self.monitors = {}
        for mod_name in self.MODULES:
            self.monitors[mod_name] = StateMonitor(self.modules[mod_name],
                                                   'v', record=True)
            # add to network to track
            self.network.add(self.monitors[mod_name])

        defaultclock.dt = dt * second

        # if reset:
        #     self.network.restore('initial')

        self.network.run(duration * second, report='text', report_period = 1*second)

    def collect_monitors(self) -> pd.DataFrame:
        """
        After a :meth:`.run` , return a dataframe of the monitored values

        Returns:
            :class:`pandas.DataFrame` - dataframe of module values over time
        """
        # make dict to construct dataframe from
        df_dict = {mod_name: np.squeeze(self.monitors[mod_name].v) for mod_name in self.MODULES}
        # and timestamps (should be same from all monitors)
        df_dict['timestamp'] = np.squeeze(self.monitors[self.MODULES[0]].t)
        return pd.DataFrame(df_dict)




    def _init_neurons(self):
        """
        Create :class:`brian2.NeuronGroup` objects for each of the modules,
        then connect them together!
        """
        # make network to keep track of created objects
        # make neuron modules!
        self.modules = {mod_name: self._make_module(mod_name) for mod_name in self.MODULES}

        # connect modules!
        # iter over nonzero positions
        for output, input in zip(*np.nonzero(self.CONNECTIVITY)):
            syn = Synapses(self.modules[self.MODULES[output]],
                           self.modules[self.MODULES[input]],
                           f'''
                           w : 1 # gap junction conductance
                           Igap_{self.MODULES[output]}_post = w * (v_pre - v_post) : 1 (summed)
                           ''')
            # just connect neurons with same index for now, should only b one
            syn.connect(j='i')
            # syn.delay = self.delay[self.MODULES[output]] * second

            # use weight, depending on connectivity matrix
            syn.w = self.weights[self.CONNECTIVITY[output, input]-1]

            # add to network to keep track of
            self.network.add(syn)

            # store in dict with key (unhashable) tuple of i/o module name)
            self.synapses[(self.modules[self.MODULES[input]],
                           self.modules[self.MODULES[output]])] = syn

        # store initial state
        # self.network.store('initial')

    def _make_module(self, name:str) -> NeuronGroup:
        eqs = '''
        dv/dt = (v0-v+Igap)/tau+sigma*xi*tau**-0.5 : 1
        v0 : 1
        tau : second
        sigma : 1 '''
        # make separate Igap equations for each possible incoming group
        subcurrents = '\n'.join([f'Igap_{mod_name} : 1' for mod_name in self.MODULES])
        # then the current used in the equation is the one that sums them all
        sum_current = 'Igap = ' + '+'.join([f'Igap_{mod_name}' for mod_name in self.MODULES]) + ' : 1'
        # then combine them all!
        eqs = '\n'.join([eqs, subcurrents, sum_current])

        G = NeuronGroup(self.size[name], eqs, method='euler')
        if self.random_init:
            G.v = 'rand()'
        G.v0 = self.resting[name]
        G.tau = self.tau[name] * second
        G.sigma = self.noise_amplitude[name]

        # add to network to track
        self.network.add(G)

        return G

    @property
    def size(self) -> typing.Dict[str, int]:
        return self._size

    @size.setter
    def size(self, size: typing.Union[int, typing.Dict[str, int]]):
        if isinstance(size, int):
            size = {k: size for k in self.MODULES}
        elif not isinstance(size, dict):
            raise ValueError(f"Dont know how to handle size, need an int or dict, got {size}")

        if any([size > 1 for size in size.values()]):
            raise NotImplementedError('Simulations with module sizes > 1 are not implemented, need to define inter-module connectivity modes to do that!')

        self._size = size

    @property
    def noise_amplitude(self) -> typing.Dict[str, float]:
        return self._noise_amplitude

    @noise_amplitude.setter
    def noise_amplitude(self, noise_amplitude: typing.Union[int, typing.Dict[str, float]]):
        if isinstance(noise_amplitude, float):
            noise_amplitude = {k: noise_amplitude for k in self.MODULES}
        elif not isinstance(noise_amplitude, dict):
            raise ValueError(f"Dont know how to handle noise_amplitude, need an int or dict, got {noise_amplitude}")
        self._noise_amplitude = noise_amplitude

    @property
    def tau(self) -> typing.Dict[str, float]:
        return self._tau

    @tau.setter
    def tau(self, tau: typing.Union[int, typing.Dict[str, float]]):
        if isinstance(tau, (float, int)):
            tau = {k: tau for k in self.MODULES}
        elif not isinstance(tau, dict):
            raise ValueError(f"Dont know how to handle tau, need an int or dict, got {tau}")
        self._tau = tau
        self._tau = tau

    @property
    def delay(self) -> typing.Dict[str, float]:
        return self._delay

    @delay.setter
    def delay(self, delay: typing.Union[int, typing.Dict[str, float]]):
        if isinstance(delay, (float, int)):
            delay = {k: delay for k in self.MODULES}
        elif not isinstance(delay, dict):
            raise ValueError(f"Dont know how to handle delay, need an int or dict, got {delay}")
        self._delay = delay
        self._delay = delay

    @property
    def resting(self) -> typing.Dict[str, float]:
        return self._resting

    @resting.setter
    def resting(self, resting: typing.Union[int, typing.Dict[str, float]]):
        if isinstance(resting, (float,int)):
            resting = {k: resting for k in self.MODULES}
        elif not isinstance(resting, dict):
            raise ValueError(f"Dont know how to handle resting, need an int or dict, got {resting}")
        self._resting = resting
        self._resting = resting

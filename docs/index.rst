.. gliske_model documentation master file, created by
   sphinx-quickstart on Sat Mar 13 22:49:02 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Lets uh simulate the dang thing
========================================

Implements the "model" in the gliske paper as a simple dynamic system.

.. image:: _images/model.png
   :width: 600
   :alt: "model"


Installation
-------------

* Clone the thing::

   git clone https://github.com/sneakers-the-rat/svg_model

* Install the dang thing::

   pip install ./svg_model

* Use the dang thing::

   from gliske.model import Gliske_Model

   mod = Gliske_Model()

   # run the model for one second
   mod.run(1)

   # get the data
   df = mod.collect_monitors()

   # lmao
   df.plot(x='timestamp', y=['distress', 'social_behavioral', 'body_ownership', 'gender', 'culture', 'feedback'])

See the :class:`gliske.model.Gliske_Model` class documentation for more specifics

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Model Docs <model>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

photonix
========

.. automodule:: photonix
   :no-members:

.. rubric:: Public names

.. autosummary::
   :nosignatures:

   ~photonix.__version__
   ~photonix.core.backend.HAS_JAX
   ~photonix.core.backend.jit
   ~photonix.core.backend.grad
   ~photonix.core.backend.value_and_grad
   ~photonix.core.backend.vmap
   ~photonix.core.backend.use_x64
   ~photonix.core.backend.backend_name
   ~photonix.core.sparams.power
   ~photonix.core.sparams.insertion_loss_db
   ~photonix.core.sparams.as_sdict
   ~photonix.core.sparams.as_sdense
   ~photonix.core.backend.to_numpy
   ~photonix.core.quality.analyze_sparameters
   ~photonix.core.quality.project_passive
   ~photonix.core.quality.SParameterDiagnostics
   ~photonix.core.spectral.differentiate_samples
   ~photonix.core.spectral.group_delay
   ~photonix.core.spectral.group_delay_dispersion
   ~photonix.linspace
   ~photonix.asarray
   ~photonix.array
   ~photonix.core.constants.WL_C_BAND
   ~photonix.core.constants.WL_DEFAULT
   ~photonix.core.dataset.SParameterDataset
   ~photonix.core.dataset.touchstone_capabilities
   ~photonix.diagnostics.runtime_info
   ~photonix.diagnostics.show_config
   ~photonix.UNAVAILABLE

.. py:data:: xp
   :module: photonix

   The module ``jax.numpy``, re-exported as ``photonix.xp`` (the active array backend: ``jax.numpy`` under JAX, ``numpy`` otherwise).

.. rubric:: Reference

.. autodata:: photonix.__version__

.. autofunction:: photonix.linspace

.. autofunction:: photonix.asarray

.. autofunction:: photonix.array

.. autodata:: photonix.UNAVAILABLE

.. rubric:: Modules

.. autosummary::
   :nosignatures:

   photonix.core
   photonix.components
   photonix.circuit
   photonix.modes
   photonix.em
   photonix.interop
   photonix.layout
   photonix.pdk
   photonix.viz
   photonix.optim
   photonix.multiphysics
   photonix.analysis
   photonix.synthesis
   photonix.cli
   photonix.diagnostics

.. toctree::
   :hidden:

   photonix.core
   photonix.components
   photonix.circuit
   photonix.modes
   photonix.em
   photonix.interop
   photonix.layout
   photonix.pdk
   photonix.viz
   photonix.optim
   photonix.multiphysics
   photonix.analysis
   photonix.synthesis
   photonix.cli
   photonix.diagnostics

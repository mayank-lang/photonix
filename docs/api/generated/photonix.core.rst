photonix.core
=============

.. automodule:: photonix.core
   :no-members:

.. rubric:: Public names

.. autosummary::
   :nosignatures:

   ~photonix.core.backend.HAS_JAX
   ~photonix.core.backend.jit
   ~photonix.core.backend.grad
   ~photonix.core.backend.value_and_grad
   ~photonix.core.backend.vmap
   ~photonix.core.backend.asarray
   ~photonix.core.backend.to_numpy
   ~photonix.core.backend.use_x64
   ~photonix.core.backend.device_count
   ~photonix.core.backend.backend_name
   ~photonix.core.types.Array
   ~photonix.core.types.Complex
   ~photonix.core.types.Float
   ~photonix.core.types.Model
   ~photonix.core.types.ModelFactory
   ~photonix.core.types.PortName
   ~photonix.core.types.PortPair
   ~photonix.core.types.SCoo
   ~photonix.core.types.SDense
   ~photonix.core.types.SDict
   ~photonix.core.types.AliasedSDict
   ~photonix.core.types.SType
   ~photonix.core.types.Settings
   ~photonix.core.dataset.SParameterDataset
   ~photonix.core.dataset.touchstone_capabilities
   ~photonix.core.quality.SParameterDiagnostics
   ~photonix.core.quality.analyze_sparameters
   ~photonix.core.quality.project_passive
   ~photonix.core.types.is_sdict
   ~photonix.core.types.is_sdense
   ~photonix.core.types.is_scoo
   ~photonix.core.types.ports_of
   ~photonix.core.sparams.as_sdict
   ~photonix.core.sparams.as_sdense
   ~photonix.core.sparams.sdict_to_sdense
   ~photonix.core.sparams.sdense_to_sdict
   ~photonix.core.sparams.reciprocal
   ~photonix.core.sparams.is_reciprocal
   ~photonix.core.sparams.is_passive
   ~photonix.core.sparams.power
   ~photonix.core.sparams.insertion_loss_db
   ~photonix.core.sparams.validate_sdict
   ~photonix.core.spectral.differentiate_samples
   ~photonix.core.spectral.group_delay
   ~photonix.core.spectral.group_delay_dispersion

.. py:data:: xp
   :module: photonix.core

   The module ``jax.numpy``, re-exported as ``photonix.core.xp`` (the active array backend: ``jax.numpy`` under JAX, ``numpy`` otherwise).

.. rubric:: Modules

.. autosummary::
   :nosignatures:

   photonix.core.constants
   photonix.core.units
   photonix.core.backend
   photonix.core.sparams
   photonix.core.quality
   photonix.core.spectral
   photonix.core.dataset
   photonix.core.types

.. toctree::
   :hidden:

   photonix.core.constants
   photonix.core.units
   photonix.core.backend
   photonix.core.sparams
   photonix.core.quality
   photonix.core.spectral
   photonix.core.dataset
   photonix.core.types

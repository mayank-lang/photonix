photonix.core.backend
=====================

.. automodule:: photonix.core.backend
   :no-members:

.. rubric:: Public names

.. autosummary::
   :nosignatures:

   ~photonix.core.backend.HAS_JAX
   ~photonix.core.backend.jit
   ~photonix.core.backend.grad
   ~photonix.core.backend.value_and_grad
   ~photonix.core.backend.vmap
   ~photonix.core.backend.jacfwd
   ~photonix.core.backend.jacrev
   ~photonix.core.backend.stop_gradient
   ~photonix.core.backend.to_numpy
   ~photonix.core.backend.asarray
   ~photonix.core.backend.use_x64
   ~photonix.core.backend.device_count
   ~photonix.core.backend.backend_name

.. py:data:: xp
   :module: photonix.core.backend

   The module ``jax.numpy``, re-exported as ``photonix.core.backend.xp`` (the active array backend: ``jax.numpy`` under JAX, ``numpy`` otherwise).

.. rubric:: Reference

.. autodata:: photonix.core.backend.HAS_JAX

.. autofunction:: photonix.core.backend.jit

.. autofunction:: photonix.core.backend.grad

.. autofunction:: photonix.core.backend.value_and_grad

.. autofunction:: photonix.core.backend.vmap

.. autofunction:: photonix.core.backend.jacfwd

.. autofunction:: photonix.core.backend.jacrev

.. autofunction:: photonix.core.backend.stop_gradient

.. autofunction:: photonix.core.backend.to_numpy

.. autofunction:: photonix.core.backend.asarray

.. autofunction:: photonix.core.backend.use_x64

.. autofunction:: photonix.core.backend.device_count

.. autofunction:: photonix.core.backend.backend_name

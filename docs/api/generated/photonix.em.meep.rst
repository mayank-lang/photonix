photonix.em.meep
================

.. automodule:: photonix.em.meep
   :no-members:

.. rubric:: Public names

.. autosummary::
   :nosignatures:

   ~photonix.em.meep.HAS_MEEP
   ~photonix.em.meep.HAS_MPB
   ~photonix.em.meep.require_meep
   ~photonix.em.meep.require_mpb
   ~photonix.em.meep.meep_frequency
   ~photonix.em.meep.n_eff_from_k
   ~photonix.em.meep.k_from_n_eff
   ~photonix.em.meep.materials.material_grid_weights
   ~photonix.em.meep.materials.index_grid
   ~photonix.em.meep.materials.epsilon_lookup
   ~photonix.em.meep.materials.to_material_grid
   ~photonix.em.meep.materials.medium
   ~photonix.em.meep.materials.to_medium
   ~photonix.em.meep.geometry.DeviceGrid
   ~photonix.em.meep.geometry.cell_size
   ~photonix.em.meep.geometry.col_to_x
   ~photonix.em.meep.geometry.row_to_y
   ~photonix.em.meep.geometry.build_block
   ~photonix.em.meep.geometry.build_pixel_block
   ~photonix.em.meep.modes.solve_modes
   ~photonix.em.meep.modes.n_eff
   ~photonix.em.meep.fdtd.waveguide_sparams
   ~photonix.em.meep.fdtd.waveguide_spectrum
   ~photonix.em.meep.fdtd.waveguide_dataset
   ~photonix.em.meep.fdtd.build_simulation
   ~photonix.em.meep.fdtd.parity_for
   ~photonix.em.meep.layout.LayerSpec
   ~photonix.em.meep.layout.PreparedPolygon
   ~photonix.em.meep.layout.PreparedPort
   ~photonix.em.meep.layout.PreparedLayout
   ~photonix.em.meep.layout.MeepLayout
   ~photonix.em.meep.layout.MeepPortRegion
   ~photonix.em.meep.layout.prepare_layout
   ~photonix.em.meep.layout.build_layout_geometry
   ~photonix.em.meep.layout.build_layout_simulation
   ~photonix.em.meep.layout.port_region
   ~photonix.em.meep.layout.port_regions
   ~photonix.em.meep.multiport.PortModeSpec
   ~photonix.em.meep.multiport.ModalTerminal
   ~photonix.em.meep.multiport.MultiportPlan
   ~photonix.em.meep.multiport.modal_terminal
   ~photonix.em.meep.multiport.plan_multiport
   ~photonix.em.meep.multiport.simulate_multiport_sparameters

.. rubric:: Reference

.. autodata:: photonix.em.meep.HAS_MEEP

.. autodata:: photonix.em.meep.HAS_MPB

.. autofunction:: photonix.em.meep.require_meep

.. autofunction:: photonix.em.meep.require_mpb

.. autofunction:: photonix.em.meep.meep_frequency

.. autofunction:: photonix.em.meep.n_eff_from_k

.. autofunction:: photonix.em.meep.k_from_n_eff

.. rubric:: Modules

.. autosummary::
   :nosignatures:

   photonix.em.meep.materials
   photonix.em.meep.geometry
   photonix.em.meep.modes
   photonix.em.meep.fdtd
   photonix.em.meep.layout
   photonix.em.meep.multiport

.. toctree::
   :hidden:

   photonix.em.meep.materials
   photonix.em.meep.geometry
   photonix.em.meep.modes
   photonix.em.meep.fdtd
   photonix.em.meep.layout
   photonix.em.meep.multiport

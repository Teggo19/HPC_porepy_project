import precice as pc
from porepyprecice import Adapter
import porepy as pp


adapter=Adapter("porepy-adapter-config.json")
#dt=1
adapter.initialize()
#adapter.advance(dt)
#adapter.finalize()
step=0

while adapter.is_coupling_ongoing():

    if adapter.requires_writing_checkpoint():
        adapter.store_checkpoint()


    precice_dt = adapter.get_max_time_step()
    pressure=adapter.read_data(precice_dt)
    step+=1
    model=adapter.make_model(north_bc_value=pressure, model_dt=step)
    pp.run_time_dependent_model(model)
    model.export_pressure_field()
    advective_flux=model.advective_flux_north_boundary()
    adapter.write_data(advective_flux)
    adapter.advance(precice_dt)

    if adapter.requires_reading_checkpoint():
        adapter.retrieve_checkpoint()


adapter.finalize()    
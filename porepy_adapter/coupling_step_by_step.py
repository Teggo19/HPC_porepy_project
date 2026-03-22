import precice as pc
from porepyprecice import Adapter

adapter=Adapter("porepy-adapter-config.json")
dt=1
adapter.initialize()
adapter.advance(dt)
adapter.finalize()

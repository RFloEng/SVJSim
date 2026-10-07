"""SVJSim — an open-source vehicle dynamics simulator for SVJ vehicles.

    from svjsim import build_vehicle, VehicleDynamics
    from svjsim.maneuvers import SlowlyIncreasingSteer
    from svj.loader import load

    sim = build_vehicle(load("car.svj.json", validate_on_load=False))
    result = SlowlyIncreasingSteer().run(VehicleDynamics(sim))
    print(result.kpi("max_lateral_g").value)
"""

from .kinematics import KCTable, StationKinematics
from .model import SimVehicle, Station, build_vehicle
from .report import write_report
from .solver import Controls, Environment, Road, State, VehicleDynamics
from .telemetry import KPI, Result
from .tires import TireModel, build_tire

__version__ = "0.5.0"

__all__ = [
    "build_vehicle", "SimVehicle", "Station",
    "VehicleDynamics", "State", "Controls", "Environment", "Road",
    "StationKinematics", "KCTable",
    "build_tire", "TireModel",
    "Result", "KPI", "write_report",
]

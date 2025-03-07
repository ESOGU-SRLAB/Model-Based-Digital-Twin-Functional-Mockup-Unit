import logging
import sys
from argparse import ArgumentParser
from concurrent import futures

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__file__)

try:
    import grpc
except ImportError:
    logger.fatal(
        "Unable to import the python library 'grpc' required by the gRPC backend. "
        "Please ensure 'grpc' is installed in the python environment launching this script. "
        "For example: 'python -m pip install unifmu[python-backend]'"
    )
    sys.exit(-1)

from schemas.unifmu_fmi2_pb2_grpc import (
    SendCommandServicer,
    add_SendCommandServicer_to_server,
    HandshakerStub,
)
from schemas.unifmu_fmi2_pb2 import (
    StatusReturn,
    GetRealReturn,
    GetIntegerReturn,
    GetBooleanReturn,
    GetStringReturn,
    HandshakeInfo,
    SerializeReturn,
    FmiStatus,
)

# Import your custom FMU class from model.py
from model import RobotFMU


class CommandServicer(SendCommandServicer):
    """
    This class implements the bridging of FMI calls (SetReal, DoStep, etc.)
    from the unifmu wrapper to your Python FMU instance. We also store
    a reference to the gRPC server, so we can stop it on Fmi2FreeInstance.
    """
    def __init__(self, fmu, server):
        super().__init__()
        logger.info("Created Python gRPC slave / backend.")
        self.fmu = fmu
        self.server = server

    # ----------------------------------------------------------------
    # Real
    # ----------------------------------------------------------------
    def Fmi2SetReal(self, request, context):
        logger.info(f"Fmi2SetReal: references={request.references}, values={request.values}")
        status = self.fmu.set_real(request.references, request.values)
        return StatusReturn(status=status)

    def Fmi2GetReal(self, request, context):
        logger.info(f"Fmi2GetReal: references={request.references}")
        values, status = self.fmu.get_real(request.references)
        return GetRealReturn(status=status, values=values)

    # ----------------------------------------------------------------
    # Integer
    # ----------------------------------------------------------------
    def Fmi2SetInteger(self, request, context):
        logger.info(f"Fmi2SetInteger: references={request.references}, values={request.values}")
        # If no integer variables are used, no special handling needed
        status = self.fmu.set_xxx(request.references, request.values)
        return StatusReturn(status=status)

    def Fmi2GetInteger(self, request, context):
        logger.info(f"Fmi2GetInteger: references={request.references}")
        status, vals = self.fmu.get_xxx(request.references)
        return GetIntegerReturn(status=status, values=vals)

    # ----------------------------------------------------------------
    # Boolean
    # ----------------------------------------------------------------
    def Fmi2SetBoolean(self, request, context):
        logger.info(f"Fmi2SetBoolean: references={request.references}, values={request.values}")
        status = self.fmu.set_xxx(request.references, request.values)
        return StatusReturn(status=status)

    def Fmi2GetBoolean(self, request, context):
        logger.info(f"Fmi2GetBoolean: references={request.references}")
        status, vals = self.fmu.get_xxx(request.references)
        return GetBooleanReturn(status=status, values=vals)

    # ----------------------------------------------------------------
    # String
    # ----------------------------------------------------------------
    def Fmi2SetString(self, request, context):
        logger.info(f"Fmi2SetString: references={request.references}, values={request.values}")
        status = self.fmu.set_xxx(request.references, request.values)
        return StatusReturn(status=status)

    def Fmi2GetString(self, request, context):
        logger.info(f"Fmi2GetString: references={request.references}")
        status, vals = self.fmu.get_xxx(request.references)
        return GetStringReturn(status=status, values=vals)

    # ----------------------------------------------------------------
    # Do Step
    # ----------------------------------------------------------------
    def Fmi2DoStep(self, request, context):
        logger.info(
            f"Fmi2DoStep: current_time={request.current_time}, "
            f"step_size={request.step_size}, no_step_prior={request.no_step_prior}"
        )
        status = self.fmu.do_step(request.current_time, request.step_size, request.no_step_prior)
        return StatusReturn(status=status)

    # ----------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------
    def Fmi2SetDebugLogging(self, request, context):
        logger.info("Fmi2SetDebugLogging called.")
        status = self.fmu.set_debug_logging(request.categories, request.logging_on)
        return StatusReturn(status=status)

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------
    def Fmi2SetupExperiment(self, request, context):
        logger.info(
            f"Fmi2SetupExperiment: start_time={request.start_time}, "
            f"stop_time={request.stop_time}, tolerance={request.tolerance}"
        )
        stop_time = request.stop_time if request.has_stop_time else None
        tol = request.tolerance if request.has_tolerance else None
        status = self.fmu.setup_experiment(request.start_time, stop_time, tol)
        return StatusReturn(status=status)

    def Fmi2EnterInitializationMode(self, request, context):
        logger.info("Fmi2EnterInitializationMode called.")
        status = self.fmu.enter_initialization_mode()
        return StatusReturn(status=status)

    def Fmi2ExitInitializationMode(self, request, context):
        logger.info("Fmi2ExitInitializationMode called.")
        status = self.fmu.exit_initialization_mode()
        return StatusReturn(status=status)

    def Fmi2Terminate(self, request, context):
        logger.info("Fmi2Terminate called.")
        status = self.fmu.terminate()
        return StatusReturn(status=status)

    def Fmi2Reset(self, request, context):
        logger.info("Fmi2Reset called.")
        status = self.fmu.reset()
        return StatusReturn(status=status)

    # ----------------------------------------------------------------
    # Free Instance
    # ----------------------------------------------------------------
    def Fmi2FreeInstance(self, request, context):
        logger.info("Fmi2FreeInstance called => stopping gRPC server.")
        # Stop the gRPC server
        self.server.stop(None)
        return StatusReturn(status=FmiStatus.Ok)

    # ----------------------------------------------------------------
    # Serialize / Deserialize
    # ----------------------------------------------------------------
    def Serialize(self, request, context):
        logger.info("Serialize called.")
        status, serialized = self.fmu.serialize()
        return SerializeReturn(status=status, state=serialized)

    def Deserialize(self, request, context):
        logger.info("Deserialize called.")
        status = self.fmu.deserialize(request.state)
        return StatusReturn(status=status)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--handshake-endpoint", dest="handshake_endpoint", type=str, required=True)
    parser.add_argument("--command-endpoint",   dest="command_endpoint",   type=str, required=False)
    args = parser.parse_args()

    # This is the handshake channel from the unifmu wrapper
    handshake_info = args.handshake_endpoint
    command_endpoint = args.command_endpoint if args.command_endpoint else "127.0.0.1:0"

    # Create the gRPC server
    import grpc
    from schemas.unifmu_fmi2_pb2_grpc import HandshakerStub
    from schemas.unifmu_fmi2_pb2 import HandshakeInfo

    server = grpc.server(futures.ThreadPoolExecutor())

    # Create your FMU instance from model.py
    fmu_instance = RobotFMU()

    # Add the command servicer with a reference to the server
    add_SendCommandServicer_to_server(CommandServicer(fmu_instance, server), server)

    # Attempt to bind to the specified ip:port
    ip, port = command_endpoint.split(":")
    port_used = server.add_insecure_port(command_endpoint)
    logger.info(f"Started FMU backend on ip={ip}, port={port_used}")

    # Start the gRPC server
    server.start()

    # Connect to the handshaker
    handshaker_channel = grpc.insecure_channel(handshake_info)
    stub = HandshakerStub(handshaker_channel)
    # Let the wrapper know which port we actually used
    hi = HandshakeInfo(ip_address=ip, port=str(port_used))
    stub.PerformHandshake(hi)
    handshaker_channel.close()

    logger.info("Handshake performed -> waiting for calls.")
    server.wait_for_termination()

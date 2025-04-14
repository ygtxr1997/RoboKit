from robokit.service.service_connector import ServiceConnector
from robokit.debug_utils.debug_classes import DebugEvaluator


gpu_connector = ServiceConnector(base_url="http://localhost:6060")
evaluator = DebugEvaluator(
    gpu_service_connector=gpu_connector,
    run_loops=200,
    conduct_actions_per_step=8,
)
evaluator.run()

from robokit.service.service_connector import ServiceConnector
from robokit.debug_utils import DebugEvaluator


gpu_connector = ServiceConnector(base_url="http://localhost:6060")
evaluator = DebugEvaluator(
    gpu_service_connector=gpu_connector,
    run_loops=400,
    conduct_actions_per_step=1,
    img_hw=(1280, 720),
)
evaluator.run()

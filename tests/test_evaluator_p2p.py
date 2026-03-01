from robokit.connects.service_connector import ServiceConnector
from robokit.debug_utils.debug_classes import DebugEvaluator


local_url = "http://localhost:6070"
remote_url = "http://g7-debug.hkueai.org"

## Option 1: Local GPU service
# test_url = local_url
## Option 2: Remote GPU service
test_url = remote_url

debug_evaluator = DebugEvaluator(
    gpu_service_connector=ServiceConnector(base_url=test_url),
    run_loops=1000,
    img_hw=(240, 320),  # (480, 848)
)
debug_evaluator.run()

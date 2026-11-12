"""Character-swap worker for the GoldenAI web-app (python -m worker --role render|util).

The web-app enqueues jobs on Redis streams (contracts/, task character_swap / character_swap_util); this package runs
them with the pipeline of this repo and reports progress and results back. See README "Web-app worker".
"""

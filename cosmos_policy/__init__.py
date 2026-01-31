######################################## for vscode debug ########################################
# uv environment will create bug when used with vscode debug
import importlib.metadata
import sys

_orig_version = importlib.metadata.version
_orig_distribution = importlib.metadata.distribution

def _patched_version(distribution_name):
    aliases = {
        "transformer_engine_torch", 
        "transformer-engine-cu12", 
        "transformer-engine-cu11",
        "transformer-engine"
    }
    if distribution_name in aliases:
        try:
            return _orig_version("transformer_engine")
        except importlib.metadata.PackageNotFoundError:
            return "2.2"
    return _orig_version(distribution_name)

importlib.metadata.version = _patched_version

def _patched_distribution(distribution_name):
    aliases = {"transformer_engine_torch", "transformer-engine-cu12", "transformer-engine"}
    if distribution_name in aliases:
        return _orig_distribution("transformer_engine")
    return _orig_distribution(distribution_name)

importlib.metadata.distribution = _patched_distribution
######################################## for vscode debug ########################################
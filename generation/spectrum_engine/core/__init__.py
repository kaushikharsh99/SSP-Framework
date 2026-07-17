"""Core types and configuration for the Spectrum Generation Engine."""

from .types import (
    PromptRecord,
    ResponseRecord,
    SamplingConfig,
    Spectrum,
    ProviderInfo,
    GenerationReport,
)
from .config import EngineConfig
from .diversity import LexicalDiversityCalculator
from .recipe import RecipeConfig, load_recipe, RecipeRegistry
from .job import GenerationJob, JobPlanner
from .composer import PromptComposer
from .executor import RecipeExecutor
from .parser import ResponseParser, ParsedResponse
from .verifier import ResponseVerifier
from .builder import SpectrumBuilder

__all__ = [
    "PromptRecord",
    "ResponseRecord",
    "SamplingConfig",
    "Spectrum",
    "ProviderInfo",
    "GenerationReport",
    "EngineConfig",
    "LexicalDiversityCalculator",
    "RecipeConfig",
    "load_recipe",
    "RecipeRegistry",
    "GenerationJob",
    "JobPlanner",
    "PromptComposer",
    "RecipeExecutor",
    "ResponseParser",
    "ParsedResponse",
    "ResponseVerifier",
    "SpectrumBuilder",
]

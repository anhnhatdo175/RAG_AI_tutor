"""Layer 5 – LLM Generation Layer (docx 2.6): structured prompting and Socratic tutoring."""
from .generator import LLMGenerator, GenerationContext
from .socratic import SocraticTutor

__all__ = ["LLMGenerator", "GenerationContext", "SocraticTutor"]

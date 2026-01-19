"""Quant Selection Agent implementation."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, Optional

from loguru import logger

from valuecell.core.agent.responses import streaming
from valuecell.core.types import BaseAgent, StreamResponse


class QuantSelectionAgent(BaseAgent):
    """Agent that explains and orchestrates the Quant Screener workflow."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the agent."""
        super().__init__(**kwargs)
        logger.info("QuantSelectionAgent initialized")

    async def stream(
        self,
        query: str,
        conversation_id: str,
        task_id: str,
        dependencies: Optional[Dict] = None,
    ) -> AsyncGenerator[StreamResponse, None]:
        """Stream a short response describing the screener capabilities."""
        logger.info(
            "QuantSelectionAgent received query: {query}",
            query=query[:100],
        )
        message = (
            "The Quant Screener scans the full U.S. universe, ranks candidates "
            "with wide + deep scores, and stores evidence/logic graphs locally. "
            "Use the Stock Screener page to run a new scan and inspect evidence "
            "chains per ticker."
        )
        for chunk in message.split(" "):
            yield streaming.message_chunk(f"{chunk} ")
        yield streaming.done()

    async def run(self, query: str, **kwargs: Any) -> str:
        """Return a short description of the screener."""
        logger.info("QuantSelectionAgent run invoked")
        return (
            "Quant Screener is ready. Run a scan from the Stock Screener page to "
            "generate candidates, evidence chains, and logic graphs."
        )

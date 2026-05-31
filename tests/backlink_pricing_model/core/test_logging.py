"""Tests for logging configuration."""

import logging

from backlink_pricing_model.core.logging import setup_logging


def test_setup_logging_returns_package_logger() -> None:
    logger = setup_logging()
    assert logger.name == "backlink_pricing_model"


def test_setup_logging_sets_level() -> None:
    logger = setup_logging(level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_setup_logging_does_not_duplicate_handlers() -> None:
    first = setup_logging()
    count = len(first.handlers)
    second = setup_logging()
    assert len(second.handlers) == count
    assert second is first

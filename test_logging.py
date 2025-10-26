#!/usr/bin/env python3
"""
Simple logging test script to verify logging setup works correctly.
"""

import os
import logging
import time


def setup_logging(log_dir):
    """Set up logging configuration - same as main.py"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"test_logging_{time.strftime('%Y%m%d_%H%M%S')}.log")

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    logging.info(f"Logging to {log_file}")
    print(f"Logging initialized - output will be saved to: {log_file}")
    return log_file


def test_logging():
    """Test logging functionality."""
    print("🔍 Testing logging setup...")

    # Setup logging
    log_file = setup_logging("./test_logs")

    # Test different log levels
    logging.info("✅ INFO level logging test")
    logging.warning("⚠️ WARNING level logging test")
    logging.error("❌ ERROR level logging test")

    # Test formatted messages
    test_value = 42
    logging.info(f"📊 Formatted message test: value = {test_value}")

    # Test multiline logging
    logging.info("🚀 Multiline test:")
    logging.info("   Line 1 of multiline message")
    logging.info("   Line 2 of multiline message")

    print(f"\n📁 Log file created at: {log_file}")

    # Check if log file exists and has content
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            content = f.read()
            print(f"📄 Log file size: {len(content)} characters")
            print("📝 Log file content preview:")
            print("-" * 50)
            print(content[-500:] if len(content) > 500 else content)
            print("-" * 50)
    else:
        print("❌ Log file was not created!")

    print("✅ Logging test completed")


if __name__ == "__main__":
    test_logging()

import logging
from logging import INFO, getLogger

import logfire

logger = getLogger(__name__)
logger.setLevel(INFO)

try:
    logfire.configure(scrubbing=False)
    logfire_handler = logfire.LogfireLoggingHandler()
    logger.addHandler(logfire_handler)

    # Disable propagation for uvicorn logs to avoid duplication
    logging.getLogger("uvicorn.access").propagate = False
    logging.getLogger("uvicorn.error").propagate = False

except Exception as e:
    # Fallback in case logfire configuration fails
    logger.setLevel(INFO)  # Ensure logger level is set if configuring Logfire fails
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(INFO)  # Set level for console output in fallback
    logger.addHandler(stream_handler)
    logger.warning(f"Failed to configure logfire: {e}")

# Optionally, for uvicorn logging levels:
# logging.getLogger("uvicorn.access").setLevel(INFO)
# logging.getLogger("uvicorn.error").setLevel(INFO)

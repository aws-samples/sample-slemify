# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Bedrock runtime client with a primary region and a fallback region.

Every Bedrock call goes to BEDROCK_REGION first (default us-west-2). If that
region cannot be reached, denies access, or reports a service error, the same
call goes once to BEDROCK_FALLBACK_REGION (default us-east-1) and a WARN line
names the failed region and the method.

The fallback covers EndpointConnectionError, ConnectTimeoutError, and the
error codes AccessDeniedException, ServiceUnavailableException and
InternalServerException. Throttling, validation and all other errors are
raised to the caller unchanged, so its own retry logic still applies. When
both variables name the same region, there is no fallback.

For converse_stream only the call that opens the stream falls back. Errors
while reading the stream reach the caller.

data-pipeline keeps a byte-for-byte copy of this file. Change both together.
"""
import logging
import os
import threading

import boto3
from botocore.exceptions import ClientError, ConnectTimeoutError, EndpointConnectionError

logger = logging.getLogger("bedrock_region")

FALLBACK_CODES = {"AccessDeniedException", "ServiceUnavailableException", "InternalServerException"}


def primary_region() -> str:
    return os.environ.get("BEDROCK_REGION") or "us-west-2"


def fallback_region() -> str:
    return os.environ.get("BEDROCK_FALLBACK_REGION") or "us-east-1"


def _should_fall_back(err: Exception) -> bool:
    if isinstance(err, (EndpointConnectionError, ConnectTimeoutError)):
        return True
    if isinstance(err, ClientError):
        return err.response.get("Error", {}).get("Code") in FALLBACK_CODES
    return False


class BedrockRuntime:
    """Two lazily created bedrock-runtime clients that share one Config."""

    def __init__(self, config=None):
        self._config = config
        self._primary = primary_region()
        self._fallback = fallback_region()
        self._clients = {}
        self._lock = threading.Lock()

    def _client(self, region):
        with self._lock:
            if region not in self._clients:
                self._clients[region] = boto3.client(
                    "bedrock-runtime", region_name=region, config=self._config)
            return self._clients[region]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._client(self._primary), name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            try:
                return getattr(self._client(self._primary), name)(*args, **kwargs)
            except Exception as err:
                if self._fallback == self._primary or not _should_fall_back(err):
                    raise
                logger.warning("Bedrock %s failed in %s (%s); calling %s instead",
                               name, self._primary, err, self._fallback)
            return getattr(self._client(self._fallback), name)(*args, **kwargs)

        return call


def bedrock_runtime(config=None) -> BedrockRuntime:
    return BedrockRuntime(config=config)

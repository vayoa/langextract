# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemini provider for LangExtract."""
# pylint: disable=duplicate-code

from __future__ import annotations

import ast
import concurrent.futures
import dataclasses
import time
from collections.abc import Mapping
from typing import Any, ClassVar, Final, Iterator, Sequence

from absl import logging

from langextract.core import base_model
from langextract.core import data
from langextract.core import exceptions
from langextract.core import schema
from langextract.core import types as core_types
from langextract.providers import patterns
from langextract.providers import router
from langextract.providers import schemas

_API_CONFIG_KEYS: Final[set[str]] = {
    'response_mime_type',
    'response_schema',
    'safety_settings',
    'system_instruction',
    'tools',
    'stop_sequences',
    'candidate_count',
}


@router.register(
    *patterns.GEMINI_PATTERNS,
    priority=patterns.GEMINI_PRIORITY,
)
@dataclasses.dataclass(init=False)
class GeminiLanguageModel(base_model.BaseLanguageModel):  # pylint: disable=too-many-instance-attributes
  """Language model inference using Google's Gemini API with structured output."""

  _RATE_LIMIT_BACKOFF_MULTIPLIER: ClassVar[float] = 1.15
  _MAX_RATE_LIMIT_RETRIES: ClassVar[int] = 5
  _MIN_RETRY_SLEEP_SECONDS: ClassVar[float] = 0.1

  model_id: str = 'gemini-2.5-flash'
  api_key: str | None = None
  vertexai: bool = False
  credentials: Any | None = None
  project: str | None = None
  location: str | None = None
  http_options: Any | None = None
  gemini_schema: schemas.gemini.GeminiSchema | None = None
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float = 0.0
  max_workers: int = 10
  fence_output: bool = False
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  @classmethod
  def get_schema_class(cls) -> type[schema.BaseSchema] | None:
    """Return the GeminiSchema class for structured output support.

    Returns:
      The GeminiSchema class that supports strict schema constraints.
    """
    return schemas.gemini.GeminiSchema

  def apply_schema(self, schema_instance: schema.BaseSchema | None) -> None:
    """Apply a schema instance to this provider.

    Args:
      schema_instance: The schema instance to apply, or None to clear.
    """
    super().apply_schema(schema_instance)
    # Keep provider behavior consistent with legacy path
    if isinstance(schema_instance, schemas.gemini.GeminiSchema):
      self.gemini_schema = schema_instance

  def __init__(
      self,
      model_id: str = 'gemini-2.5-flash',
      api_key: str | None = None,
      vertexai: bool = False,
      credentials: Any | None = None,
      project: str | None = None,
      location: str | None = None,
      http_options: Any | None = None,
      gemini_schema: schemas.gemini.GeminiSchema | None = None,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float = 0.0,
      max_workers: int = 10,
      fence_output: bool = False,
      **kwargs,
  ) -> None:
    """Initialize the Gemini language model.

    Args:
      model_id: The Gemini model ID to use.
      api_key: API key for Gemini service.
      vertexai: Whether to use Vertex AI instead of API key authentication.
      credentials: Optional Google auth credentials for Vertex AI.
      project: Google Cloud project ID for Vertex AI.
      location: Vertex AI location (e.g., 'global', 'us-central1').
      http_options: Optional HTTP options for the client (e.g., for VPC endpoints).
      gemini_schema: Optional schema for structured output.
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      max_workers: Maximum number of parallel API calls.
      fence_output: Whether to wrap output in markdown fences (ignored,
        Gemini handles this based on schema).
      **kwargs: Additional Gemini API parameters. Only allowlisted keys are
        forwarded to the API (response_schema, response_mime_type, tools,
        safety_settings, stop_sequences, candidate_count, system_instruction).
        See https://ai.google.dev/api/generate-content for details.
    """
    try:
      # pylint: disable=import-outside-toplevel
      from google import genai
    except ImportError as e:
      raise exceptions.InferenceConfigError(
          'google-genai is required for Gemini. Install it with: pip install'
          ' google-genai'
      ) from e

    self.model_id = model_id
    self.api_key = api_key
    self.vertexai = vertexai
    self.credentials = credentials
    self.project = project
    self.location = location
    self.http_options = http_options
    self.gemini_schema = gemini_schema
    self.format_type = format_type
    self.temperature = temperature
    self.max_workers = max_workers
    self.fence_output = fence_output

    if not self.api_key and not self.vertexai:
      raise exceptions.InferenceConfigError(
          'Gemini models require either:\n  - An API key via api_key parameter'
          ' or LANGEXTRACT_API_KEY env var\n  - Vertex AI configuration with'
          ' vertexai=True, project, and location'
      )
    if self.vertexai and (not self.project or not self.location):
      raise exceptions.InferenceConfigError(
          'Vertex AI mode requires both project and location parameters'
      )

    if self.api_key and self.vertexai:
      logging.warning(
          'Both API key and Vertex AI configuration provided. '
          'API key will take precedence for authentication.'
      )

    self._client = genai.Client(
        api_key=self.api_key,
        vertexai=vertexai,
        credentials=credentials,
        project=project,
        location=location,
        http_options=http_options,
    )

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )
    self._extra_kwargs = {
        k: v for k, v in (kwargs or {}).items() if k in _API_CONFIG_KEYS
    }

  def _process_single_prompt(
      self, prompt: str, config: dict
  ) -> core_types.ScoredOutput:
    """Process a single prompt and return a ScoredOutput."""
    for attempt in range(self._MAX_RATE_LIMIT_RETRIES + 1):
      try:
        # Apply stored kwargs that weren't already set in config
        for key, value in self._extra_kwargs.items():
          if key not in config and value is not None:
            config[key] = value

        if self.gemini_schema:
          # Structured output requires JSON format
          if self.format_type != data.FormatType.JSON:
            raise exceptions.InferenceConfigError(
                'Gemini structured output only supports JSON format. '
                'Set format_type=JSON or use_schema_constraints=False.'
            )
          config.setdefault('response_mime_type', 'application/json')
          config.setdefault('response_schema', self.gemini_schema.schema_dict)

        response = self._client.models.generate_content(
            model=self.model_id, contents=prompt, config=config
        )

        return core_types.ScoredOutput(score=1.0, output=response.text)

      except Exception as e:  # pylint: disable=broad-except
        wait_seconds = self._extract_minute_rate_limit_delay(e)
        should_retry = (
            wait_seconds is not None and attempt < self._MAX_RATE_LIMIT_RETRIES
        )
        if should_retry:
          sleep_seconds = max(
              wait_seconds * self._RATE_LIMIT_BACKOFF_MULTIPLIER,
              self._MIN_RETRY_SLEEP_SECONDS,
          )
          logging.warning(
              'Gemini minute rate limit hit; retrying in %.2fs (attempt %d/%d).',
              sleep_seconds,
              attempt + 1,
              self._MAX_RATE_LIMIT_RETRIES,
          )
          time.sleep(sleep_seconds)
          continue

        raise exceptions.InferenceRuntimeError(
            f'Gemini API error: {str(e)}', original=e
        ) from e

  def _extract_minute_rate_limit_delay(self, error: Exception) -> float | None:
    client_error = self._unwrap_client_error(error)
    if client_error is None:
      return None

    status_code = getattr(client_error, 'status_code', None)
    if status_code != 429:
      status_code = getattr(client_error, 'code', None)
      if status_code != 429:
        return None

    payload = self._extract_error_payload(client_error)
    if not isinstance(payload, Mapping):
      return None

    details = payload.get('details', [])
    if not isinstance(details, Sequence) or isinstance(details, (str, bytes)):
      return None
    if not self._has_minute_quota_violation(details):
      return None

    retry_value = self._find_retry_delay(details)
    if retry_value is None:
      retry_value = payload.get('retryDelay') or payload.get('retry_delay')
    if retry_value is None:
      retry_value = getattr(client_error, 'retry_delay', None)

    return self._coerce_retry_delay_seconds(retry_value)

  def _unwrap_client_error(self, error: Exception) -> Exception | None:
    if self._is_client_error_instance(error):
      return error
    cause = getattr(error, '__cause__', None)
    if isinstance(cause, Exception) and self._is_client_error_instance(cause):
      return cause
    context = getattr(error, '__context__', None)
    if isinstance(context, Exception) and self._is_client_error_instance(context):
      return context
    return None

  def _is_client_error_instance(self, error: Exception) -> bool:
    return (
        error.__class__.__name__ == 'ClientError'
        and error.__class__.__module__ == 'google.genai.errors'
    )

  def _extract_error_payload(self, error: Exception) -> Mapping[str, Any] | None:
    # Prefer structured payloads attached to the error when available.
    response_json = getattr(error, 'response_json', None)
    if isinstance(response_json, Mapping):
      nested_error = response_json.get('error')
      if isinstance(nested_error, Mapping):
        return nested_error
      return response_json

    for arg in getattr(error, 'args', ()):  # pragma: no branch
      if isinstance(arg, Mapping):
        nested_error = arg.get('error') if hasattr(arg, 'get') else None
        if isinstance(nested_error, Mapping):
          return nested_error
        return arg

    text = str(error)
    brace_index = text.find('{')
    if brace_index == -1:
      return None
    try:
      parsed = ast.literal_eval(text[brace_index:])
    except (SyntaxError, ValueError):  # pragma: no cover - defensive
      return None
    if isinstance(parsed, Mapping):
      nested_error = parsed.get('error') if hasattr(parsed, 'get') else None
      if isinstance(nested_error, Mapping):
        return nested_error
      return parsed
    return None

  def _has_minute_quota_violation(self, details: Sequence[Any]) -> bool:
    for detail in details:
      if not isinstance(detail, Mapping):
        continue
      if not detail.get('@type', '').endswith('QuotaFailure'):
        continue
      violations = detail.get('violations') or []
      if not isinstance(violations, Sequence) or isinstance(violations, (str, bytes)):
        continue
      for violation in violations:
        if not isinstance(violation, Mapping):
          continue
        quota_id = violation.get('quotaId') or violation.get('quota_id')
        if isinstance(quota_id, str) and 'PerMinute' in quota_id:
          return True
    return False

  def _find_retry_delay(self, details: Sequence[Any]) -> Any:
    for detail in details:
      if not isinstance(detail, Mapping):
        continue
      if not detail.get('@type', '').endswith('RetryInfo'):
        continue
      retry_delay = detail.get('retryDelay')
      if retry_delay is None:
        retry_delay = detail.get('retry_delay')
      if retry_delay is not None:
        return retry_delay
    return None

  def _coerce_retry_delay_seconds(self, value: Any) -> float | None:
    if value is None:
      return None
    if isinstance(value, (int, float)):
      return max(float(value), 0.0)
    if isinstance(value, str):
      trimmed = value.strip().lower()
      if trimmed.endswith('s'):
        trimmed = trimmed[:-1]
      try:
        return max(float(trimmed), 0.0)
      except ValueError:
        return None
    if isinstance(value, Mapping):
      seconds = value.get('seconds')
      nanos = value.get('nanos')
      if seconds is None and nanos is None:
        return None
      seconds_value = float(seconds) if seconds is not None else 0.0
      nanos_value = float(nanos) if nanos is not None else 0.0
      return max(seconds_value + nanos_value / 1_000_000_000, 0.0)
    return None

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[core_types.ScoredOutput]]:
    """Runs inference on a list of prompts via Gemini's API.

    Args:
      batch_prompts: A list of string prompts.
      **kwargs: Additional generation params (temperature, top_p, top_k, etc.)

    Yields:
      Lists of ScoredOutputs.
    """
    merged_kwargs = self.merge_kwargs(kwargs)

    config = {
        'temperature': merged_kwargs.get('temperature', self.temperature),
    }
    if 'max_output_tokens' in merged_kwargs:
      config['max_output_tokens'] = merged_kwargs['max_output_tokens']
    if 'top_p' in merged_kwargs:
      config['top_p'] = merged_kwargs['top_p']
    if 'top_k' in merged_kwargs:
      config['top_k'] = merged_kwargs['top_k']

    handled_keys = {'temperature', 'max_output_tokens', 'top_p', 'top_k'}
    for key, value in merged_kwargs.items():
      if (
          key not in handled_keys
          and key in _API_CONFIG_KEYS
          and value is not None
      ):
        config[key] = value

    # Use parallel processing for batches larger than 1
    if len(batch_prompts) > 1 and self.max_workers > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(self.max_workers, len(batch_prompts))
      ) as executor:
        future_to_index = {
            executor.submit(
                self._process_single_prompt, prompt, config.copy()
            ): i
            for i, prompt in enumerate(batch_prompts)
        }

        results: list[core_types.ScoredOutput | None] = [None] * len(
            batch_prompts
        )
        for future in concurrent.futures.as_completed(future_to_index):
          index = future_to_index[future]
          try:
            results[index] = future.result()
          except Exception as e:
            raise exceptions.InferenceRuntimeError(
                f'Parallel inference error: {str(e)}', original=e
            ) from e

        for result in results:
          if result is None:
            raise exceptions.InferenceRuntimeError(
                'Failed to process one or more prompts'
            )
          yield [result]
    else:
      # Sequential processing for single prompt or worker
      for prompt in batch_prompts:
        result = self._process_single_prompt(prompt, config.copy())
        yield [result]  # pylint: disable=duplicate-code

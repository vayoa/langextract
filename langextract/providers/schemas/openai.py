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

"""Schema implementation for OpenAI structured JSON output."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Any
import warnings

from langextract.core import data
from langextract.core import format_handler as fh
from langextract.core import schema


def _build_extraction_schema(
    examples_data: Sequence[data.ExampleData],
    attribute_suffix: str,
) -> dict[str, Any]:
  """Helper that converts ExampleData into a JSON schema fragment."""

  extraction_categories: dict[str, dict[str, set[type]]] = {}
  for example in examples_data:
    for extraction in example.extractions:
      category = extraction.extraction_class
      if category not in extraction_categories:
        extraction_categories[category] = {}

      if extraction.attributes:
        for attr_name, attr_value in extraction.attributes.items():
          attr_types = extraction_categories[category].setdefault(
              attr_name, set()
          )
          attr_types.add(type(attr_value))

  extraction_properties: dict[str, dict[str, Any]] = {}

  for category, attrs in extraction_categories.items():
    extraction_properties[category] = {"type": "string"}

    attributes_field = f"{category}{attribute_suffix}"
    attr_properties: dict[str, Any] = {}

    if not attrs:
      attr_properties["_unused"] = {"type": "string"}
    else:
      for attr_name, attr_types in attrs.items():
        if list in attr_types:
          attr_properties[attr_name] = {
              "type": "array",
              "items": {"type": "string"},
          }
        else:
          attr_properties[attr_name] = {"type": "string"}

    extraction_properties[attributes_field] = {
        "type": "object",
        "properties": attr_properties,
        "nullable": True,
    }

  return {
      "type": "object",
      "properties": extraction_properties,
  }


@dataclasses.dataclass
class OpenAISchema(schema.BaseSchema):
  """Schema implementation for OpenAI's json_schema response format."""

  _schema_dict: dict[str, Any]
  name: str = "LangExtractSchema"
  strict: bool = True

  @property
  def schema_dict(self) -> dict[str, Any]:
    return self._schema_dict

  def to_provider_config(self) -> dict[str, Any]:
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "schema": self._schema_dict,
            },
            "strict": self.strict,
        }
    }

  @property
  def requires_raw_output(self) -> bool:
    return True

  def validate_format(self, format_handler: fh.FormatHandler) -> None:
    if format_handler.use_fences:
      warnings.warn(
          "OpenAI json_schema responses return raw JSON. Disable fence_output.",
          UserWarning,
          stacklevel=3,
      )

    if (
        not format_handler.use_wrapper
        or format_handler.wrapper_key != data.EXTRACTIONS_KEY
    ):
      warnings.warn(
          "OpenAI structured output expects wrapper_key='extractions'.",
          UserWarning,
          stacklevel=3,
      )

  @classmethod
  def from_examples(
      cls,
      examples_data: Sequence[data.ExampleData],
      attribute_suffix: str = data.ATTRIBUTE_SUFFIX,
  ) -> OpenAISchema:
    extraction_schema = _build_extraction_schema(examples_data, attribute_suffix)

    schema_dict = {
        "type": "object",
        "properties": {
            data.EXTRACTIONS_KEY: {
                "type": "array",
                "items": extraction_schema,
            }
        },
        "required": [data.EXTRACTIONS_KEY],
    }

    return cls(_schema_dict=schema_dict)

  def sync_with_provider_kwargs(self, kwargs: dict[str, Any]) -> None:
    response_format = kwargs.get("response_format")
    if not response_format:
      return
    if response_format.get("type") != "json_schema":
      return
    json_schema = response_format.get("json_schema", {})
    if name := json_schema.get("name"):
      self.name = name
    if schema_dict := json_schema.get("schema"):
      self._schema_dict = schema_dict
    if "strict" in response_format:
      self.strict = bool(response_format["strict"])

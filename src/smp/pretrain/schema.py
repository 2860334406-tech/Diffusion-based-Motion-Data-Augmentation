"""Explicit checkpoint schemas; NULL IDs must never leak across model versions."""

from smp.pretrain.conditioning import CONDITION_SCHEMA as FIVE_SCHEMA
from smp.pretrain.shoulder80 import CONDITION_SCHEMA as THREE_SCHEMA
from smp.pretrain.shoulder_ik_v2 import CONDITION_SCHEMA as IK_V2_SCHEMA
from smp.pretrain.shoulder_mixed import CONDITION_SCHEMA as MIXED_SCHEMA


def validate_schema(schema, num_conditions):
  expected = {3: (THREE_SCHEMA, IK_V2_SCHEMA, MIXED_SCHEMA), 5: (FIVE_SCHEMA,)}.get(
    num_conditions, ()
  )
  if not any(schema == candidate for candidate in expected):
    raise ValueError("Conditional prior dictionary/version mismatch")
  return schema


def resolve_condition(value, schema):
  validate_schema(schema, len(schema["names"]))
  if isinstance(value, str):
    if value == "null":
      return schema["null_id"]
    if value in schema["names"]:
      return schema["names"].index(value)
    if not value.isdecimal():
      raise ValueError(f"Condition {value!r} is absent from this checkpoint schema")
    value = int(value)
  if type(value) is not int or not 0 <= value <= schema["null_id"]:
    raise ValueError("Invalid condition ID")
  return value

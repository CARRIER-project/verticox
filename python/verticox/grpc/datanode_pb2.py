# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: verticox/grpc/datanode.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1cverticox/grpc/datanode.proto\")\n\rUpdateRequest\x12\t\n\x01z\x18\x01 \x03(\x02\x12\r\n\x05gamma\x18\x02 \x03(\x02\"0\n\x10LocalAuxiliaries\x12\r\n\x05gamma\x18\x01 \x03(\x02\x12\r\n\x05sigma\x18\x02 \x03(\x02\"\"\n\x0bNumFeatures\x12\x13\n\x0bnumFeatures\x18\x01 \x01(\x05\"\x07\n\x05\x45mpty2c\n\x08\x44\x61taNode\x12-\n\x06update\x12\x0e.UpdateRequest\x1a\x11.LocalAuxiliaries\"\x00\x12(\n\x0egetNumFeatures\x12\x06.Empty\x1a\x0c.NumFeatures\"\x00\x62\x06proto3')



_UPDATEREQUEST = DESCRIPTOR.message_types_by_name['UpdateRequest']
_LOCALAUXILIARIES = DESCRIPTOR.message_types_by_name['LocalAuxiliaries']
_NUMFEATURES = DESCRIPTOR.message_types_by_name['NumFeatures']
_EMPTY = DESCRIPTOR.message_types_by_name['Empty']
UpdateRequest = _reflection.GeneratedProtocolMessageType('UpdateRequest', (_message.Message,), {
  'DESCRIPTOR' : _UPDATEREQUEST,
  '__module__' : 'verticox.grpc.datanode_pb2'
  # @@protoc_insertion_point(class_scope:UpdateRequest)
  })
_sym_db.RegisterMessage(UpdateRequest)

LocalAuxiliaries = _reflection.GeneratedProtocolMessageType('LocalAuxiliaries', (_message.Message,), {
  'DESCRIPTOR' : _LOCALAUXILIARIES,
  '__module__' : 'verticox.grpc.datanode_pb2'
  # @@protoc_insertion_point(class_scope:LocalAuxiliaries)
  })
_sym_db.RegisterMessage(LocalAuxiliaries)

NumFeatures = _reflection.GeneratedProtocolMessageType('NumFeatures', (_message.Message,), {
  'DESCRIPTOR' : _NUMFEATURES,
  '__module__' : 'verticox.grpc.datanode_pb2'
  # @@protoc_insertion_point(class_scope:NumFeatures)
  })
_sym_db.RegisterMessage(NumFeatures)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), {
  'DESCRIPTOR' : _EMPTY,
  '__module__' : 'verticox.grpc.datanode_pb2'
  # @@protoc_insertion_point(class_scope:Empty)
  })
_sym_db.RegisterMessage(Empty)

_DATANODE = DESCRIPTOR.services_by_name['DataNode']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _UPDATEREQUEST._serialized_start=32
  _UPDATEREQUEST._serialized_end=73
  _LOCALAUXILIARIES._serialized_start=75
  _LOCALAUXILIARIES._serialized_end=123
  _NUMFEATURES._serialized_start=125
  _NUMFEATURES._serialized_end=159
  _EMPTY._serialized_start=161
  _EMPTY._serialized_end=168
  _DATANODE._serialized_start=170
  _DATANODE._serialized_end=269
# @@protoc_insertion_point(module_scope)

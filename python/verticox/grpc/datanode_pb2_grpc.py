# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from verticox.grpc import datanode_pb2 as verticox_dot_grpc_dot_datanode__pb2


class DataNodeStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.fit = channel.unary_unary(
                '/DataNode/fit',
                request_serializer=verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
                response_deserializer=verticox_dot_grpc_dot_datanode__pb2.LocalParameters.FromString,
                )
        self.getNumFeatures = channel.unary_unary(
                '/DataNode/getNumFeatures',
                request_serializer=verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
                response_deserializer=verticox_dot_grpc_dot_datanode__pb2.NumFeatures.FromString,
                )
        self.getNumSamples = channel.unary_unary(
                '/DataNode/getNumSamples',
                request_serializer=verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
                response_deserializer=verticox_dot_grpc_dot_datanode__pb2.NumSamples.FromString,
                )
        self.updateParameters = channel.unary_unary(
                '/DataNode/updateParameters',
                request_serializer=verticox_dot_grpc_dot_datanode__pb2.AggregatedParameters.SerializeToString,
                response_deserializer=verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
                )
        self.computeGamma = channel.unary_unary(
                '/DataNode/computeGamma',
                request_serializer=verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
                response_deserializer=verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
                )
        self.getBeta = channel.unary_unary(
                '/DataNode/getBeta',
                request_serializer=verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
                response_deserializer=verticox_dot_grpc_dot_datanode__pb2.Beta.FromString,
                )
        self.prepare = channel.unary_unary(
                '/DataNode/prepare',
                request_serializer=verticox_dot_grpc_dot_datanode__pb2.InitialValues.SerializeToString,
                response_deserializer=verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
                )
        self.kill = channel.unary_unary(
                '/DataNode/kill',
                request_serializer=verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
                response_deserializer=verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
                )
        self.getFeatureNames = channel.unary_unary(
                '/DataNode/getFeatureNames',
                request_serializer=verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
                response_deserializer=verticox_dot_grpc_dot_datanode__pb2.FeatureNames.FromString,
                )
        self.getAverageSigma = channel.unary_unary(
                '/DataNode/getAverageSigma',
                request_serializer=verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
                response_deserializer=verticox_dot_grpc_dot_datanode__pb2.AverageSigma.FromString,
                )
        self.getRecordLevelSigma = channel.unary_unary(
                '/DataNode/getRecordLevelSigma',
                request_serializer=verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
                response_deserializer=verticox_dot_grpc_dot_datanode__pb2.RecordLevelSigma.FromString,
                )
        self.getPartialHazardRatio = channel.unary_unary(
                '/DataNode/getPartialHazardRatio',
                request_serializer=verticox_dot_grpc_dot_datanode__pb2.Subset.SerializeToString,
                response_deserializer=verticox_dot_grpc_dot_datanode__pb2.PartialHazardRatio.FromString,
                )


class DataNodeServicer(object):
    """Missing associated documentation comment in .proto file."""

    def fit(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getNumFeatures(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getNumSamples(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def updateParameters(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def computeGamma(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getBeta(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def prepare(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def kill(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getFeatureNames(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getAverageSigma(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getRecordLevelSigma(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getPartialHazardRatio(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DataNodeServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'fit': grpc.unary_unary_rpc_method_handler(
                    servicer.fit,
                    request_deserializer=verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
                    response_serializer=verticox_dot_grpc_dot_datanode__pb2.LocalParameters.SerializeToString,
            ),
            'getNumFeatures': grpc.unary_unary_rpc_method_handler(
                    servicer.getNumFeatures,
                    request_deserializer=verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
                    response_serializer=verticox_dot_grpc_dot_datanode__pb2.NumFeatures.SerializeToString,
            ),
            'getNumSamples': grpc.unary_unary_rpc_method_handler(
                    servicer.getNumSamples,
                    request_deserializer=verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
                    response_serializer=verticox_dot_grpc_dot_datanode__pb2.NumSamples.SerializeToString,
            ),
            'updateParameters': grpc.unary_unary_rpc_method_handler(
                    servicer.updateParameters,
                    request_deserializer=verticox_dot_grpc_dot_datanode__pb2.AggregatedParameters.FromString,
                    response_serializer=verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
            ),
            'computeGamma': grpc.unary_unary_rpc_method_handler(
                    servicer.computeGamma,
                    request_deserializer=verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
                    response_serializer=verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
            ),
            'getBeta': grpc.unary_unary_rpc_method_handler(
                    servicer.getBeta,
                    request_deserializer=verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
                    response_serializer=verticox_dot_grpc_dot_datanode__pb2.Beta.SerializeToString,
            ),
            'prepare': grpc.unary_unary_rpc_method_handler(
                    servicer.prepare,
                    request_deserializer=verticox_dot_grpc_dot_datanode__pb2.InitialValues.FromString,
                    response_serializer=verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
            ),
            'kill': grpc.unary_unary_rpc_method_handler(
                    servicer.kill,
                    request_deserializer=verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
                    response_serializer=verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
            ),
            'getFeatureNames': grpc.unary_unary_rpc_method_handler(
                    servicer.getFeatureNames,
                    request_deserializer=verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
                    response_serializer=verticox_dot_grpc_dot_datanode__pb2.FeatureNames.SerializeToString,
            ),
            'getAverageSigma': grpc.unary_unary_rpc_method_handler(
                    servicer.getAverageSigma,
                    request_deserializer=verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
                    response_serializer=verticox_dot_grpc_dot_datanode__pb2.AverageSigma.SerializeToString,
            ),
            'getRecordLevelSigma': grpc.unary_unary_rpc_method_handler(
                    servicer.getRecordLevelSigma,
                    request_deserializer=verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
                    response_serializer=verticox_dot_grpc_dot_datanode__pb2.RecordLevelSigma.SerializeToString,
            ),
            'getPartialHazardRatio': grpc.unary_unary_rpc_method_handler(
                    servicer.getPartialHazardRatio,
                    request_deserializer=verticox_dot_grpc_dot_datanode__pb2.Subset.FromString,
                    response_serializer=verticox_dot_grpc_dot_datanode__pb2.PartialHazardRatio.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'DataNode', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class DataNode(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def fit(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DataNode/fit',
            verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
            verticox_dot_grpc_dot_datanode__pb2.LocalParameters.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getNumFeatures(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DataNode/getNumFeatures',
            verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
            verticox_dot_grpc_dot_datanode__pb2.NumFeatures.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getNumSamples(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DataNode/getNumSamples',
            verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
            verticox_dot_grpc_dot_datanode__pb2.NumSamples.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def updateParameters(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DataNode/updateParameters',
            verticox_dot_grpc_dot_datanode__pb2.AggregatedParameters.SerializeToString,
            verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def computeGamma(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DataNode/computeGamma',
            verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
            verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getBeta(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DataNode/getBeta',
            verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
            verticox_dot_grpc_dot_datanode__pb2.Beta.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def prepare(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DataNode/prepare',
            verticox_dot_grpc_dot_datanode__pb2.InitialValues.SerializeToString,
            verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def kill(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DataNode/kill',
            verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
            verticox_dot_grpc_dot_datanode__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getFeatureNames(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DataNode/getFeatureNames',
            verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
            verticox_dot_grpc_dot_datanode__pb2.FeatureNames.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getAverageSigma(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DataNode/getAverageSigma',
            verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
            verticox_dot_grpc_dot_datanode__pb2.AverageSigma.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getRecordLevelSigma(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DataNode/getRecordLevelSigma',
            verticox_dot_grpc_dot_datanode__pb2.Empty.SerializeToString,
            verticox_dot_grpc_dot_datanode__pb2.RecordLevelSigma.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getPartialHazardRatio(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/DataNode/getPartialHazardRatio',
            verticox_dot_grpc_dot_datanode__pb2.Subset.SerializeToString,
            verticox_dot_grpc_dot_datanode__pb2.PartialHazardRatio.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

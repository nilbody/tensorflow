#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_COMPRESSION_HELPER_H
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_COMPRESSION_HELPER_H

namespace grpc {
    class ByteBuffer;
}

void Compression(grpc::ByteBuffer *data);

void Decompression(grpc::ByteBuffer *data);

#endif
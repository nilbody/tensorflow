#include <iostream>
#include <map>   //BTBT MAYBUG:change to unorder_map ?
#include <atomic>
#include <list>
#include <mutex>
#include "grpc++/support/byte_buffer.h"
#include "grpc++/support/slice.h"
#include "tensorflow/core/distributed_runtime/rpc/compression_helper.h"
#include "tensorflow/core/distributed_runtime/rpc/lz4.h"

static const int32_t kMinDataSize = 1024 * 1024;
static const int32_t kMaxDataSize = 128 * 1024 * 1024; //BTBT ???: int32_t

//make Magic longer to reduce conflict opportunity
static const char kLz4Magic[4] = {0x18, 0x4d, 0x22, 0x04};

class BufferPool {
    static const int32_t kAlign = 1 << 22; //BTBT ???
public:
    char* Allocate(int32_t len) {
        int32_t inner_len = (len + kAlign - 1) / kAlign * kAlign;
        char *ptr = nullptr;
        {
            std::lock_guard<std::mutex> lck(mutex_);
            auto iter = pool_.find(inner_len);
            if(iter != pool_.end() && !iter->second.empty()) {
                ptr = iter->second.front();
                iter->second.pop_front();
            }
            if(!ptr) {
                ptr = new char[inner_len];
                current_capacity_ += inner_len;
                if(current_capacity_ < capacity_) {
                    info_[ptr] = inner_len;
                }
            }
        }
        return ptr;
    }

    void Deallocate(void *ptr) {
        char *data = reinterpret_cast<char*>(ptr);
        std::lock_guard<std::mutex> lck(mutex_);
        auto iter = info_.find(data);
        if(iter != info_.end()) {
            pool_[iter->second].push_back(data);
        } else {
            delete [] data;
        }
    }

    static BufferPool* Instance() {
        static std::mutex mutex;
        static BufferPool *instance = nullptr;
        if(!instance) {
            std::lock_guard<std::mutex> lck(mutex);
            if(!instance) {
                BufferPool *tmp = new BufferPool();
                instance = tmp;
            }
        }
        return instance;
    }

private:
    BufferPool(uint64_t capacity = 8 * 1024 * 1024 * 1024 1024UL)
        : capacity_(capacity), current_capacity_(0) {
        }
    
    ~BufferPool() {
        for (auto &item : info_) {
            delete [] item.first;
        }
    }
private:
    std::mutex mutex_;
    std::map<char*, int32_t> info_;
    std::map<int32_t, std::list<char*>> pool_;
    u_int64_t capacity_;
    std::atomic<u_int64_t> current_capacity_;
};

void Destroy(void *ptr) {
    BufferPool::Instance()->Deallocate(ptr); //BTBT MAYBUG: 这样貌似达不到单例模式吧
}

void Compression(grpc::ByteBuffer *data) {
    static auto f = [](bool v) {
        std::cout << "TF_COMPRESSION : " << v << std::endl; //BTBT REFACTOR: 加上个人域
        return v;
    };
    static bool use_compression = getenv("TF_COMPRESSION") ? f(true) : f(false);
    if (use_compression) {
        static std::atomic<u_int64_t> before_compress_size{0};
        static std::atomic<u_int64_t> after_compress_size{0};
        static std::atomic<u_int64_t> cnt{0};
        static std::atomic<u_int64_t> uncompress_size{0};
        int32_t data_len = data->length():
        if(data_len < kMinDataSize || data_len > kMaxDataSize) {
            uncompress_size += data_len; //BTBT MAYBUG: why add? just for log? but it will become larger and larger.
            return;
        }
        char *input = BufferPool::Instance()->Allocate(data_len);
        char *output = BufferPool::Instance()->Allocate(data_len + sizeof(kLz4Magic));
        std::vector<grpc::Slice> slices;
        data->Dump(&slices);
        int32_t len = 0;
        for(const auto &slice: slices) {
            memcpy(input + len, slice.begin, slice.size());
            len += slice.size();
        }

        int dst_len = LZ4_compress_default(input, output + sizeof(kLz4Magic), 
                                            len, data_len + sizeof(kLz4Magic));
        uint64_t count = ++cnt;
        if(dst_len > 0 && dst_len < len) {
            for(uint32_t ix = 0; ix < sizseof(kLz4Magic); ix++) {
                output[ix] = kLz4Magic[ix];
            }
            grpc::Slice slice(output, dst_len + sizseof(kLz4Magic), Destroy);
            grpc::ByteBuffer result(&slice, 1);
            data->Swap(&result);
            before_compress_sizse += len;
            after_compress_size += dst_len;
        } else {
            Destroy(output);
            uncompress_size += data_len;
        }
        Destroy(input);
        if((count % 1000) == 0) {
            std::cout << "compress_count : " << count << ", before_compress : "
                << before_compress_size << ", after_compress : " << after_compress_size
                << ", uncompress : " << uncompress_size << std::endl;
        }
    }
}

void Decompression(grpc::ByteBuffer *data) {
    static bool use_compression = getenv("TF_COMPRESSION") ? true : false;
    if(use_compression) {
        int32_t data_len = data->length();
        if(data_len < sizeof(kLz4Magic)) {
            return;
        }
        std::vector<grpc::Slice> slices;
        data->Dump(&slices);
        const char *ref = reinterpret_cast<const char*>(slices[0].begin());
        for(uint32_t ix = 0; ix < sizeof(kLz4Magic); ix++) {
            if(ref[ix] != kLz4Magic[ix]) {
                return;
            }
        }
        char *input = BufferPool::Instance()->Allocate(data_len);
        char *output = BufferPool::Instance()->Allocate(kMaxDataSize);
        int32_t len = 0;
        for(const auto &slice: slices) {
            memcpy(input + len, slice.begin(), slice.size());
            len += slice.size();
        }

        int dst_len = LZ4_decompress_safe(input + sizeof(kLz4Magic), output,
                                            len - sizeof(kLz4Magic), kMaxDataSize);
        if(dst_len <= 0 || dst_len < len) {
            Destroy(input);
            Destroy(output);
            return;
        }
        grpc::Slice slice(output, dst_len, Destroy);
        grpc::ByteBuffer result(&slice, 1);
        data->Swap(&result);
        Destroy(input);
    }
}





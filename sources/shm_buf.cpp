#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include "bytes_buffer.hpp"
#include "shm_buf.h"
using namespace boost::interprocess;
mapped_region *s_region = 0;


SharedMemoryBuffer::SharedMemoryBuffer(const char *shm_name, uint32_t bufsize, bool delete_shm)
{
    _shm_name = shm_name;
    _bufsize = bufsize;
    _delete_shm = delete_shm;
    if (!init_shm())
        std::cerr << "init_shm failed!" << std::endl;
}

SharedMemoryBuffer::~SharedMemoryBuffer()
{   
    if (s_region)
        delete s_region;

    if (_delete_shm)
        shared_memory_object::remove(_shm_name);
}

void SharedMemoryBuffer::remove_shm(const char *shm_name)
{
    shared_memory_object::remove(shm_name);
}

bool SharedMemoryBuffer::init_shm()
{
    try
    {
        shared_memory_object shm;
        try
        {
            shm = shared_memory_object(open_only, _shm_name, read_write);
            boost::interprocess::offset_t size;
            shm.get_size(size);
            _bufsize = (uint32_t)size;
        }
        catch(const std::exception& e)
        {
            if (_bufsize > 0)
            {
                shm = shared_memory_object(create_only, _shm_name, read_write);
                shm.truncate(_bufsize);
            }
            else
            {
                std::cerr << "open or create shm " << _shm_name << "failed!" << std::endl;
                return false;
            }
        }
        boost::interprocess::offset_t size;
        if (shm.get_size(size))
           std::cout << shm.get_name() << ", size:" << size << std::endl;
        if (size == 0)
            return false;

        // Map the whole shared memory in this process
        s_region = new mapped_region(shm, read_write);
        _bytesBuf = new BytesBuffer((char *)s_region->get_address(), _bufsize);
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "exception: " << e.what() << std::endl;
        return false;
    }
}

bool SharedMemoryBuffer::write_shm(const char *data, uint32_t len)
{
    bool ret = false;
    ret = _bytesBuf->append((char *)&len, 4);
    if (!ret)
        return false;
    ret = _bytesBuf->append(data, len);
    if (!ret)
        return false;
    return true;
}

bool SharedMemoryBuffer::readable()
{
    return _bytesBuf->readable_size() > 0 ? true : false;
}

// 
uint32_t SharedMemoryBuffer::read_shm(std::string& data) {
    uint32_t len = 0;
    printf("Step 1: Start read_shm\n");

    // 检查是否有可读数据
    if (readable()) {
        printf("Step 2: Data is readable\n");

        // 从缓冲区中读取前4个字节，假定它们表示数据的长度
        _bytesBuf->retrieve((char *)&len, 4);

        // 检查读取的数据长度是否合理
        if (len > 0 && len <= _bytesBuf->readable_size()) {
            printf("Step 3: Length of data is %u\n", len);

            // 调整 data 字符串的大小以容纳即将读取的数据
            data.resize(len);

            // 从缓冲区中读取实际数据
            len = _bytesBuf->retrieve((char *)data.data(), len);
            printf("Step 4: Successfully read %u bytes of data\n", len);

            return len;  // 返回读取的数据长度
        } else {
            // 数据长度不合理，可能是读取过程中出现了问题
            printf("Step 3: Invalid length %u\n", len);
            return 0;
        }
    } else {
        printf("Step 2: No data to read\n");
        return 0;
    }
}

void SharedMemoryBuffer::clear_buffer() {
    if (_bytesBuf) {
        _bytesBuf->clear();
    }
}
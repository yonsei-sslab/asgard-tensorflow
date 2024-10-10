// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "include/rknn/1.6.0/rknn_api.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/resource.h>
#include <linux/vm_sockets.h>

#define BUF_SIZE 64
#define MEM_SIZE 224 * 224 * 3

#define IOCTL_ATTACH_DEVICE _IO('w', 0x00)
#define IOCTL_DETACH_DEVICE _IO('w', 0x01)

/*-------------------------------------------
                  Functions
-------------------------------------------*/
static inline unsigned long long get_timer_counter(void)
{
  unsigned long long val;
  asm volatile("isb; mrs %0, cntpct_el0" : "=r"(val));
  return val;
}

static int rknn_GetTopN(float* pfProb, float* pfMaxProb, uint32_t* pMaxClass, uint32_t outputCount, uint32_t topNum)
{
  uint32_t i, j;
  uint32_t top_count = outputCount > topNum ? topNum : outputCount;

  for (i = 0; i < topNum; ++i) {
    pfMaxProb[i] = -FLT_MAX;
    pMaxClass[i] = -1;
  }

  for (j = 0; j < top_count; j++) {
    for (i = 0; i < outputCount; i++) {
      if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
          (i == *(pMaxClass + 4))) {
        continue;
      }

      if (pfProb[i] > *(pfMaxProb + j)) {
        *(pfMaxProb + j) = pfProb[i];
        *(pMaxClass + j) = i;
      }
    }
  }

  return 1;
}

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE* fp = fopen(filename, "rb");
  if (fp == nullptr) {
    printf("fopen %s fail!\n", filename);
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  int            model_len = ftell(fp);
  unsigned char* model     = (unsigned char*)malloc(model_len);
  fseek(fp, 0, SEEK_SET);
  if (model_len != fread(model, 1, model_len, fp)) {
    printf("fread %s fail!\n", filename);
    free(model);
    return NULL;
  }
  *model_size = model_len;
  if (fp) {
    fclose(fp);
  }
  return model;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char* argv[])
{
  if (argc < 1) {
    return -1;
  }

  char* model_path = argv[1];

  // Set up vsock server
  int s = socket(AF_VSOCK, SOCK_STREAM, 0);
  if (s < 0) {
    perror("socket");
    exit(errno);
  }

  int vfd = open("/dev/vsock", O_RDWR);
  if (vfd < 0) {
    perror("open");
    exit(errno);
  }

  unsigned int cid;
  int ret = ioctl(vfd, IOCTL_VM_SOCKETS_GET_LOCAL_CID, &cid);
  if (ret < 0) {
    perror("ioctl");
    exit(errno);
  }

  struct sockaddr_vm addr;
  memset(&addr, 0, sizeof(struct sockaddr_vm));
  addr.svm_family = AF_VSOCK;
  addr.svm_port = 9999;
  addr.svm_cid = cid;

  ret = bind(s, (const sockaddr*) &addr, sizeof(struct sockaddr_vm));
  if (ret < 0) {
    perror("bind");
    exit(errno);
  }

  ret = listen(s, 0);
  if (ret < 0) {
    perror("listen");
    exit(errno);
  }

  int ioctl_fd = open("/dev/hypcall", O_RDWR);
  if (ioctl_fd < 0) {
    perror("open");
    exit(errno);
  }

  // Attach RKNPU
  int ioctl_ret = ioctl(ioctl_fd, IOCTL_ATTACH_DEVICE);
  if (ioctl_ret) {
    printf("attach device failed\n");
  }

  rknn_context ctx = 0;

  // Load RKNN Model
  int            model_len = 0;
  unsigned char* model     = load_model(model_path, &model_len);
  ret = rknn_init(&ctx, model, model_len, 0, NULL);
  if (ret < 0) {
    printf("rknn_init fail! ret=%d\n", ret);
    return -1;
  }

  // Get sdk and driver version
  rknn_sdk_version sdk_ver;
  ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }

  printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);

  // Get Model Input Output Info
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }

  printf("input tensors:\n");
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
  for (uint32_t i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0) {
      printf("rknn_init error! ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&input_attrs[i]);
  }

  printf("output tensors:\n");
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
  for (uint32_t i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&output_attrs[i]);
  }

  // Get custom string
  rknn_custom_string custom_string;
  ret = rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }

  // Create input tensor memory
  rknn_tensor_mem* input_mems[1];
  input_attrs[0].type = RKNN_TENSOR_UINT8;
  input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride * 2);

  // Create output tensor memory
  rknn_tensor_mem* output_mems[io_num.n_output];
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    // default output type is depend on model, this require float32 to compute top5
    // allocate float32 output tensor
    int output_size = output_attrs[i].n_elems * sizeof(float);
    output_mems[i]  = rknn_create_mem(ctx, output_size);
  }

  // Set input tensor memory
  ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
  if (ret < 0) {
    printf("rknn_set_io_mem fail! ret=%d\n", ret);
    return -1;
  }

  // Set output tensor memory
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    // set output memory and attribute
    ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
    if (ret < 0) {
      printf("rknn_set_io_mem fail! ret=%d\n", ret);
      return -1;
    }
  }

  // Open and prepared shared memory device
  int mem_fd = open("/dev/io-mem", O_RDWR | O_SYNC);
  if (mem_fd < 0) {
    perror("open");
    exit(errno);
  }

  char* mem = (char*) mmap(NULL, MEM_SIZE + 4096, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, 0);
  if (mem == MAP_FAILED) {
    perror("mmap");
    exit(errno);
  }

  char* buf = (char*) malloc(BUF_SIZE);
  if (buf == NULL) {
      perror("malloc");
      exit(errno);
  }

  // Set nice value to the highest priority (-20)
  errno = 0;
  setpriority(PRIO_PROCESS, 0, -20);
  if (errno) {
    perror("setpriority");
    exit(errno);
  }

  // Bind the current process to cpu0
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(0, &mask);
  ret = sched_setaffinity(0, sizeof(mask), &mask);
  if (ret < 0) {
    perror("sched_setaffinity");
    exit(errno);
  }

  // Detach RKNPU
  ioctl_ret = ioctl(ioctl_fd, IOCTL_DETACH_DEVICE);
  if (ioctl_ret) {
    perror("detach device failed");
    exit(errno);
  }

  socklen_t peer_addr_size = sizeof(struct sockaddr_vm);
  while (1) {
    struct sockaddr_vm peer_addr;
    int peer_fd = accept(s, (sockaddr*) &peer_addr, &peer_addr_size);
    if (peer_fd < 0) {
      perror("accept");
      exit(errno);
    }

    size_t msg_len;
    while ((msg_len = recv(peer_fd, buf, BUF_SIZE, 0)) > 0) {
      // Attach RKNPU
      ioctl_ret = ioctl(ioctl_fd, IOCTL_ATTACH_DEVICE);
      if (ioctl_ret) {
        printf("attach device failed\n");
      }

      unsigned long long time_1 = get_timer_counter();

      // Copy input data to input tensor memory
      int width = input_attrs[0].dims[2];
      int stride = input_attrs[0].w_stride;

      if (width == stride) {
        memcpy(input_mems[0]->virt_addr, mem, width * input_attrs[0].dims[1] * input_attrs[0].dims[3]);
      } else {
        int height = input_attrs[0].dims[1];
        int channel = input_attrs[0].dims[3];
        // copy from src to dst with stride
        uint8_t *src_ptr = (uint8_t *)mem;
        uint8_t *dst_ptr = (uint8_t *)input_mems[0]->virt_addr;
        // width-channel elements
        int src_wc_elems = width * channel;
        int dst_wc_elems = stride * channel;
        for (int h = 0; h < height; ++h) {
          memcpy(dst_ptr, src_ptr, src_wc_elems);
          src_ptr += src_wc_elems;
          dst_ptr += dst_wc_elems;
        }
      }

      unsigned long long time_2 = get_timer_counter();

      // Run
      ret = rknn_run(ctx, NULL);
      if (ret < 0) {
        printf("rknn run error %d\n", ret);
        return -1;
      }

      unsigned long long time_3 = get_timer_counter();

      // DEBUG: Print top 5
      uint32_t topNum = 5;
      uint32_t MaxClass[topNum];
      for (uint32_t i = 0; i < io_num.n_output; i++) {
        float    fMaxProb[topNum];
        float*   buffer    = (float*)output_mems[i]->virt_addr;
        uint32_t sz        = output_attrs[i].n_elems;
        int      top_count = sz > topNum ? topNum : sz;

        rknn_GetTopN(buffer, fMaxProb, MaxClass, sz, topNum);

        printf("---- Top%d ----\n", top_count);
        for (int j = 0; j < top_count; j++) {
          printf("%8.6f - %d\n", fMaxProb[j], MaxClass[j]);
        }
      }

      // Copy output into shared device
      memcpy(mem + MEM_SIZE, &MaxClass[0], sizeof(uint32_t));
      msync(mem, MEM_SIZE + sizeof(uint32_t), 0);

      unsigned long long time_4 = get_timer_counter();

      // Detach RKNPU
      ioctl_ret = ioctl(ioctl_fd, IOCTL_DETACH_DEVICE);
      if (ioctl_ret) {
        perror("detach device failed");
        exit(errno);
      }

      // Send message back to vm_service
      snprintf(buf, BUF_SIZE, "%llu,%llu,%llu", time_3 - time_2, time_1, time_4);
      send(peer_fd, (void*) buf, BUF_SIZE, 0);
    }
  }

  // Attach RKNPU
  ioctl_ret = ioctl(ioctl_fd, IOCTL_ATTACH_DEVICE);
  if (ioctl_ret) {
    printf("attach device failed\n");
    exit(errno);
  }

  // Destroy rknn memory
  rknn_destroy_mem(ctx, input_mems[0]);
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    rknn_destroy_mem(ctx, output_mems[i]);
  }

  // destroy
  rknn_destroy(ctx);

  // Detach RKNPU
  ioctl_ret = ioctl(ioctl_fd, IOCTL_DETACH_DEVICE);
  if (ioctl_ret) {
    perror("detach device failed");
    exit(errno);
  }

  if (model != nullptr) {
    free(model);
  }

  munmap(mem, MEM_SIZE + 4096);
  close(mem_fd);
  close(ioctl_fd);
  close(vfd);

  return 0;
}

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
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <linux/types.h>
#include <drm/drm.h>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "include/stb/stb_image_resize.h"

#define BUF_SIZE 64
#define MEM_SIZE 0x4000000

#define RKNPU_GET_DRV_COMP_PERF_NUMS 0x08
#define RKNPU_CLEAR_DRV_COMP_PERF_NUMS 0x09
#define RKNPU_SET_DRV_COMP_PERF_NUMS 0x10
#define DRM_COMMAND_BASE 0x40
#define DRM_IOCTL_RKNPU_GET_DRV_COMP_PERF_NUMS                                 \
	DRM_IOWR(DRM_COMMAND_BASE + RKNPU_GET_DRV_COMP_PERF_NUMS, struct rknpu_drv_comp_perf_eval)
#define DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS                               \
	DRM_IO(DRM_COMMAND_BASE + RKNPU_CLEAR_DRV_COMP_PERF_NUMS)
#define DRM_IOCTL_RKNPU_SET_DRV_COMP_PERF_NUMS                                 \
	DRM_IOWR(DRM_COMMAND_BASE + RKNPU_SET_DRV_COMP_PERF_NUMS, struct rknpu_drv_comp_perf_eval)

struct rknpu_drv_comp_perf_eval {
	__u32 cnt_limit;
	__u64 cnt_total;
};

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

static float* load_image_float(const char* image_path)
{
  int req_channel = 3;
  int height  = 0;
  int width   = 0;
  int channel = 0;

  float* image_data = stbi_loadf(image_path, &width, &height, &channel, req_channel);
  if (image_data == NULL) {
    printf("load image failed!\n");
    return NULL;
  }

  return image_data;
}

static unsigned char* load_image_int(const char* image_path)
{
  int req_channel = 3;
  int height  = 0;
  int width   = 0;
  int channel = 0;

  unsigned char* image_data = stbi_load(image_path, &width, &height, &channel, req_channel);
  if (image_data == NULL) {
    printf("load image failed!\n");
    return NULL;
  }

  return image_data;
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
  char* input_path = argv[1];
  char* sock_path = argv[2];
  char* mem_path = argv[3];
  int num_rknn_models = atoi(argv[4]);

  int loop_count = 1;
  if (argc > (num_rknn_models + 6)) {
    loop_count = atoi(argv[num_rknn_models + 6]);
  }

  if (argc < (num_rknn_models + 6)) {
    printf("Usage:%s input_path socket_path shared_memory_path num_rknn_models rknn_models... [loop_count]\n", argv[0]);
    return -1;
  }

  // Setup RKNN runtime
  rknn_context ctx[num_rknn_models];
  int model_len[num_rknn_models];
  rknn_input_output_num io_num[num_rknn_models];
  char** rknn_model_path = (char**) malloc(sizeof(char*) * num_rknn_models);
  unsigned char** rknn_model = (unsigned char**) malloc(sizeof(unsigned char*) * num_rknn_models);

  rknn_tensor_attr** input_attrs = (rknn_tensor_attr**) malloc(sizeof(rknn_tensor_attr*) * num_rknn_models);
  rknn_tensor_attr** output_attrs = (rknn_tensor_attr**) malloc(sizeof(rknn_tensor_attr*) * num_rknn_models);
  rknn_tensor_mem*** input_mems = (rknn_tensor_mem***) malloc(sizeof(rknn_tensor_mem**) * num_rknn_models);
  rknn_tensor_mem*** output_mems = (rknn_tensor_mem***) malloc(sizeof(rknn_tensor_mem**) * num_rknn_models);

  unsigned char* input_data = NULL;
  int ret;

  for (int i = 0; i < num_rknn_models; i++) {
    // Initialize ctx
    rknn_model_path[i] = argv[i + 5];
    rknn_model[i] = load_model(rknn_model_path[i], &model_len[i]);
    ret = rknn_init(&ctx[i], rknn_model[i], model_len[i], 0, NULL);
    if (ret < 0) {
      printf("rknn_init fail! ret=%d\n", ret);
      return -1;
    }

    // Query io_num
    ret = rknn_query(ctx[i], RKNN_QUERY_IN_OUT_NUM, &io_num[i], sizeof(io_num[i]));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }

    // Query input_attrs
    printf("input tensors:\n");
    input_attrs[i] = (rknn_tensor_attr*) malloc(sizeof(rknn_tensor_attr) * io_num[i].n_input);
    memset(input_attrs[i], 0, io_num[i].n_input * sizeof(rknn_tensor_attr));
    for (uint32_t j = 0; j < io_num[i].n_input; j++) {
      input_attrs[i][j].index = j;
      ret = rknn_query(ctx[i], RKNN_QUERY_INPUT_ATTR, &(input_attrs[i][j]), sizeof(rknn_tensor_attr));
      if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
      }
      dump_tensor_attr(&input_attrs[i][j]);
    }

    // Query output_attrs
    printf("output tensors:\n");
    output_attrs[i] = (rknn_tensor_attr*) malloc(sizeof(rknn_tensor_attr) * io_num[i].n_output);
    memset(output_attrs[i], 0, io_num[i].n_output * sizeof(rknn_tensor_attr));
    for (uint32_t j = 0; j < io_num[i].n_output; j++) {
      output_attrs[i][j].index = j;
      ret = rknn_query(ctx[i], RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i][j]), sizeof(rknn_tensor_attr));
      if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
      }
      dump_tensor_attr(&output_attrs[i][j]);
    }

    // Create input tensor memory
    if (i == 0) {
      input_mems[0] = (rknn_tensor_mem**) malloc(sizeof(rknn_tensor_mem*) * io_num[i].n_input);
      input_attrs[0][0].type = RKNN_TENSOR_UINT8;
      input_mems[0][0] = rknn_create_mem(ctx[0], input_attrs[0][0].size_with_stride * 2);
    } else {
      input_mems[i] = (rknn_tensor_mem**) malloc(sizeof(rknn_tensor_mem*) * io_num[i].n_input);
      for (uint32_t j = 0; j < io_num[i].n_input; ++j) {
        input_mems[i][j] = rknn_create_mem(ctx[i], input_attrs[i][j].size_with_stride);
      }
    }

    // Create output tensor memory
    output_mems[i] = (rknn_tensor_mem**) malloc(sizeof(rknn_tensor_mem*) * io_num[i].n_output);
    for (uint32_t j = 0; j < io_num[i].n_output; ++j) {
      int output_size = output_attrs[i][j].n_elems * sizeof(float);
      output_mems[i][j] = rknn_create_mem(ctx[i], output_size);
    }

    // Set input tensor memory
    for (uint32_t j = 0; j < io_num[i].n_input; ++j) {
      ret = rknn_set_io_mem(ctx[i], input_mems[i][j], &input_attrs[i][j]);
      if (ret < 0) {
        printf("rknn_set_io_mem fail! ret=%d\n", ret);
        return -1;
      }
    }

    // Set output tensor memory
    for (uint32_t j = 0; j < io_num[i].n_output; ++j) {
      // set output memory and attribute
      ret = rknn_set_io_mem(ctx[i], output_mems[i][j], &output_attrs[i][j]);
      if (ret < 0) {
        printf("rknn_set_io_mem fail! ret=%d\n", ret);
        return -1;
      }
    }
  }

  // Get sdk and driver version
  rknn_sdk_version sdk_ver;
  ret = rknn_query(ctx[0], RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);

  // Get custom string
  rknn_custom_string custom_string;
  ret = rknn_query(ctx[0], RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  printf("custom string: %s\n", custom_string.string);

  // Set up socket to VM_SERVICE
  int sfd = socket(PF_LOCAL, SOCK_STREAM, 0);
  if (sfd < 0) {
    perror("socket");
    exit(errno);
  }

  struct sockaddr_un my_addr;
  memset(&my_addr, 0, sizeof(my_addr));
  my_addr.sun_family = AF_LOCAL;
  strncpy(my_addr.sun_path, sock_path, sizeof(my_addr.sun_path) - 1);

  ret = connect(sfd, (struct sockaddr *) &my_addr, sizeof(my_addr));
  if (ret < 0) {
    perror("connect");
    exit(errno);
  }

  // Open output file
  FILE* output_file = fopen("output.csv", "w+");
  if (output_file == NULL) {
    perror("fopen");
    exit(errno);
  }

  // Open RKNPU DRM device
  int rknpu_fd = open("/dev/dri/card0", O_RDWR);
  if (rknpu_fd < 0) {
      perror("open");
      exit(errno);
  }

  // Set nice value to the highest priority (-20)
  errno = 0;
  setpriority(PRIO_PROCESS, 0, -20);
  if (errno) {
    perror("setpriority");
    exit(errno);
  }

  fprintf(output_file, "iteration,inference total,inference ree,inference tee\n");

  // Bind the current process to cpu7
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(7, &mask);
  ret = sched_setaffinity(0, sizeof(mask), &mask);
  if (ret < 0) {
    perror("sched_setaffinity");
    exit(errno);
  }

  char* buf = (char*) malloc(BUF_SIZE);
  if (buf == NULL) {
      perror("malloc");
      exit(errno);
  }

  printf("Begin running %i inferences\n", loop_count);
  for (int cnt = 0; cnt < loop_count; cnt++) {
    // Do not measure time for open() and mmap()
    int fd = open(mem_path, O_RDWR | O_SYNC);
    if (fd < 0) {
      perror("open");
      exit(errno);
    }

    char* mem = (char*) mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED) {
      perror("mmap");
      exit(errno);
    }

    // Set NPU computation count limit
    struct rknpu_drv_comp_perf_eval drv_comp_perf_eval;
    drv_comp_perf_eval.cnt_limit = 100;
    ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_SET_DRV_COMP_PERF_NUMS, &drv_comp_perf_eval);
    if (ret) {
        perror("ioctl");
        exit(errno);
    }

    // Clear NPU computation time
    ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
    if (ret) {
        perror("ioctl");
        exit(errno);
    }

    // Load image
    input_data = load_image_int(input_path);
    if (!input_data) {
      return -1;
    }

    // Copy input data to input tensor memory
    int width  = input_attrs[0][0].dims[2];
    int stride = input_attrs[0][0].w_stride;

    if (width == stride) {
      memcpy(input_mems[0][0]->virt_addr, input_data, width * input_attrs[0][0].dims[1] * input_attrs[0][0].dims[3]);
    } else {
      int height  = input_attrs[0][0].dims[1];
      int channel = input_attrs[0][0].dims[3];
      // copy from src to dst with stride
      uint8_t* src_ptr = input_data;
      uint8_t* dst_ptr = (uint8_t*)input_mems[0][0]->virt_addr;
      // width-channel elements
      int src_wc_elems = width * channel;
      int dst_wc_elems = stride * channel;
      for (int h = 0; h < height; ++h) {
        memcpy(dst_ptr, src_ptr, src_wc_elems);
        src_ptr += src_wc_elems;
        dst_ptr += dst_wc_elems;
      }
    }

    // Print measurement
    fprintf(output_file, "%i,", cnt);

    unsigned long long vsock_send_end_time, deobfuscate_time, relu_time, obfuscate_time, vsock_recv_start_time;
    unsigned long long ree_time = 0;
    unsigned long long tee_time = 0;
    unsigned long long vsock_time = 0;
    for (int j = 0; j < num_rknn_models-1; j++) {
      // Run
      ret = rknn_run(ctx[j], NULL);
      if (ret < 0) {
        printf("rknn run error %d\n", ret);
        return -1;
      }

      // Get NPU computation time
      struct rknpu_drv_comp_perf_eval drv_comp_perf_eval;
      ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_GET_DRV_COMP_PERF_NUMS, &drv_comp_perf_eval);
      if (ret) {
          perror("ioctl");
          exit(errno);
      }

      // Clear NPU computation time
      ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
      if (ret) {
          perror("ioctl");
          exit(errno);
      }

      // Print measurement
      // fprintf(output_file, "%llu,", drv_comp_perf_eval.cnt_total);
      ree_time = ree_time + drv_comp_perf_eval.cnt_total;

      memcpy(mem, output_mems[j][0]->virt_addr, output_mems[j][0]->size);

      ret = msync(mem, output_mems[j][0]->size, 0);
      if (ret < 0) {
        perror("msync");
        exit(errno);
      }

      unsigned long long vsock_send_start_time = get_timer_counter();

      // Send inference request
      ret = send(sfd, "start inference", 15, 0);
      if (ret < 0) {
        perror("send");
        exit(errno);
      }

      // Wait for VM_SERVICE to reply
      size_t msg_len = recv(sfd, buf, BUF_SIZE, 0);
      if (ret < 0) {
        perror("recv");
        exit(errno);
      }

      unsigned long long vsock_recv_end_time = get_timer_counter();

      // Print measurements
      sscanf(buf, "%llu,%llu,%llu,%llu,%llu",
                  &vsock_send_end_time, &deobfuscate_time, &relu_time, &obfuscate_time, &vsock_recv_start_time);
      // fprintf(output_file, "%llu,%llu,%llu,%llu,%llu,",
      //                      vsock_send_end_time - vsock_send_start_time,
      //                      deobfuscate_time, relu_time, obfuscate_time,
      //                      vsock_recv_end_time - vsock_recv_start_time);
      tee_time = tee_time + deobfuscate_time + relu_time + obfuscate_time;
      vsock_time = vsock_time + vsock_send_end_time - vsock_send_start_time;
      vsock_time = vsock_time + vsock_recv_end_time - vsock_recv_start_time;

      memset(buf, 0, BUF_SIZE);

      memcpy(input_mems[j+1][0]->virt_addr, mem, input_mems[j+1][0]->size);
    }

    unsigned long long last_time = get_timer_counter();

    // Run the last fragment
    ret = rknn_run(ctx[num_rknn_models-1], NULL);
    if (ret < 0) {
      printf("rknn run error %d\n", ret);
      return -1;
    }

    last_time = get_timer_counter() - last_time;
    ree_time = ree_time + last_time;

    // Print measurement
    // NOTE: The timer counter must be running at 24 MHz
    fprintf(output_file, "%.2f\n", (ree_time + tee_time + vsock_time) / 24000.0);

    // Get top 5
    uint32_t topNum = 5;
    for (uint32_t i = 0; i < io_num[num_rknn_models-1].n_output; i++) {
      uint32_t MaxClass[topNum];
      float    fMaxProb[topNum];
      float*   buffer    = (float*)output_mems[num_rknn_models-1][i]->virt_addr;
      uint32_t sz        = output_attrs[num_rknn_models-1][i].n_elems;
      int      top_count = sz > topNum ? topNum : sz;

      rknn_GetTopN(buffer, fMaxProb, MaxClass, sz, topNum);

      printf("---- Top%d ----\n", top_count);
      for (int j = 0; j < top_count; j++) {
        printf("%8.6f - %d\n", fMaxProb[j], MaxClass[j]);
      }
    }
    printf("Iteration: %d/%d\n", cnt + 1, loop_count);

    munmap(mem, MEM_SIZE);
    close(fd);
  }

  close(sfd);
  fclose(output_file);

  return 0;
}

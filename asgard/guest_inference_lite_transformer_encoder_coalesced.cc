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

#include <memory>
#include <iostream>
#include <ostream>
#include <queue>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

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
#include <drm/drm.h>
#include <linux/vm_sockets.h>

#define BUF_SIZE 64
#define MEM_SIZE 224 * 224 * 3

#define IOCTL_ATTACH_DEVICE _IO('w', 0x00)
#define IOCTL_DETACH_DEVICE _IO('w', 0x01)

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

typedef TfLiteFloat16 float16_t;

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
  int num_rknn_models = 16;
  int num_tfl_models = 10;

  if (argc < (num_rknn_models + num_tfl_models)) {
    printf("Usage:%s rknn_models... tfl_models...\n", argv[0]);
    return -1;
  }

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

  for (int i = 0; i < num_rknn_models; i++) {
    // Initialize ctx
    rknn_model_path[i] = argv[i + 1];

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
      int output_size = output_attrs[i][j].n_elems * 2;
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

  // Setup TFLite runtime
  char** tfl_model_path = (char**) malloc(sizeof(char*) * num_tfl_models);
  std::unique_ptr<tflite::FlatBufferModel>* tfl_model = new std::unique_ptr<tflite::FlatBufferModel>[num_tfl_models];
  tflite::ops::builtin::BuiltinOpResolver* tfl_resolver = new tflite::ops::builtin::BuiltinOpResolver[num_tfl_models];
  std::unique_ptr<tflite::Interpreter>* tfl_interpreter = new std::unique_ptr<tflite::Interpreter>[num_tfl_models];

  for (int i = 0; i < num_tfl_models; i++) {
    tfl_model_path[i] = argv[i + num_rknn_models + 1];

    // Load the model
    tfl_model[i] = tflite::FlatBufferModel::BuildFromFile(tfl_model_path[i]);
    if (!tfl_model[i]) {
      printf("Failed to mmap model\n");
    }

    // Build the interpreter
    tflite::InterpreterBuilder(*tfl_model[i], tfl_resolver[i])(&tfl_interpreter[i]);
    if (!tfl_interpreter[i]){
      printf("Failed to construct interpreter\n");
    }

    tfl_interpreter[i]->SetNumThreads(1);
    if (tfl_interpreter[i]->AllocateTensors() != kTfLiteOk) {
      printf("Failed to allocate tensors!\n");
    }
  }

  // Open RKNPU DRM device
  int rknpu_fd = open("/dev/dri/card0", O_RDWR);
  if (rknpu_fd < 0) {
      perror("open");
      exit(errno);
  }

  // Open and prepared shared memory device
  int mem_fd = open("/dev/io-mem", O_RDWR | O_SYNC);
  if (mem_fd < 0) {
    perror("open");
    exit(errno);
  }

  unsigned char* mem = (unsigned char*) mmap(NULL, MEM_SIZE + 4096, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, 0);
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

      // Fill input buffers with all 2s
      memset(input_mems[0][0]->virt_addr, 2, input_mems[0][0]->size);
      memset(input_mems[1][0]->virt_addr, 2, input_mems[1][0]->size);

      int rknn_index = 0;
      int tfl_index = 0;
      float* tfl_input_tensors[2];

      unsigned long long npu_time = 0;
      unsigned long long cpu_time = 0;
      unsigned long long time_1, time_2;
      struct rknpu_drv_comp_perf_eval drv_comp_perf_eval;

      // Set NPU computation count limit
      drv_comp_perf_eval.cnt_limit = 1;
      ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_SET_DRV_COMP_PERF_NUMS, &drv_comp_perf_eval);
      if (ret) {
          perror("ioctl");
          exit(errno);
      }

      // Run encoder layer 1

      // Clear NPU computation time
      ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
      if (ret) {
          perror("ioctl");
          exit(errno);
      }

      // c1-1: Run RKNN inference
      ret = rknn_run(ctx[rknn_index], NULL);
      if (ret < 0) {
        printf("rknn run error %d\n", ret);
        return -1;
      }

      // Get NPU computation time
      ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_GET_DRV_COMP_PERF_NUMS, &drv_comp_perf_eval);
      if (ret) {
          perror("ioctl");
          exit(errno);
      }

      npu_time = npu_time + drv_comp_perf_eval.cnt_total;

      // Clear NPU computation time
      ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
      if (ret) {
          perror("ioctl");
          exit(errno);
      }

      // c1-2: Run RKNN inference
      rknn_index++;
      ret = rknn_run(ctx[rknn_index], NULL);
      if (ret < 0) {
        printf("rknn run error %d\n", ret);
        return -1;
      }

      // Get NPU computation time
      ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_GET_DRV_COMP_PERF_NUMS, &drv_comp_perf_eval);
      if (ret) {
          perror("ioctl");
          exit(errno);
      }

      npu_time = npu_time + drv_comp_perf_eval.cnt_total;

      // c1-3: Copy intermediate results
      for (int i = 0; i < 2; ++i) {
          tfl_input_tensors[i] = tfl_interpreter[tfl_index]->typed_input_tensor<float>(i);
          if (!tfl_input_tensors[i]) {
              printf("Failed to construct tfl_input_tensor c1-3 %d\n", i);
          }
      }
      memcpy(tfl_input_tensors[0], output_mems[rknn_index][1]->virt_addr, output_mems[rknn_index][1]->size);
      memcpy(tfl_input_tensors[1], output_mems[rknn_index][0]->virt_addr, output_mems[rknn_index][0]->size);

      time_1 = get_timer_counter();

      // c1-3: Run TFLite inference
      tfl_interpreter[tfl_index]->Invoke();

      time_2 = get_timer_counter();
      cpu_time = cpu_time + (time_2 - time_1);

      // c1-4: Copy intermediate results
      rknn_index++;
      memcpy(input_mems[rknn_index][0]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float>(0), input_mems[rknn_index][0]->size);         // Copy from c1-3 fragment (tfl_index: 0)
      memcpy(input_mems[rknn_index][1]->virt_addr, output_mems[0][0]->virt_addr, output_mems[0][0]->size);                                              // Copy from c1-1 fragment (rknn_index: 0)
      memcpy(input_mems[rknn_index][2]->virt_addr, output_mems[0][1]->virt_addr, output_mems[0][1]->size);                                              // Copy from c1-1 fragment (rknn_index: 0)

      // Clear NPU computation time
      ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
      if (ret) {
          perror("ioctl");
          exit(errno);
      }

      // c1-4: Run RKNN inference
      ret = rknn_run(ctx[rknn_index], NULL);
      if (ret < 0) {
        printf("rknn run error %d\n", ret);
        return -1;
      }

      // Get NPU computation time
      ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_GET_DRV_COMP_PERF_NUMS, &drv_comp_perf_eval);
      if (ret) {
          perror("ioctl");
          exit(errno);
      }

      npu_time = npu_time + drv_comp_perf_eval.cnt_total;

      // c1-5: Copy intermediate results
      tfl_index++;
      for (int i = 0; i < 2; ++i) {
          tfl_input_tensors[i] = tfl_interpreter[tfl_index]->typed_input_tensor<float>(i);
          if (!tfl_input_tensors[i]) {
              printf("Failed to construct tfl_input_tensor %d\n", i);
          }
      }
      memcpy(tfl_input_tensors[0], output_mems[rknn_index][0]->virt_addr, output_mems[rknn_index][0]->size);         // Copy from c1-4 fragment (rknn_index: 2)
      memcpy(tfl_input_tensors[1], output_mems[rknn_index-1][2]->virt_addr, output_mems[rknn_index-1][2]->size);     // Copy from c1-2 fragment (rknn_index: 1)

      time_1 = get_timer_counter();

      // c1-5: Run TFLite inference
      tfl_interpreter[tfl_index]->Invoke();

      time_2 = get_timer_counter();
      cpu_time = cpu_time + (time_2 - time_1);

      // c1-6: Copy intermediate results
      rknn_index++;
      memcpy(input_mems[rknn_index][0]->virt_addr, output_mems[rknn_index-2][3]->virt_addr, output_mems[rknn_index-2][3]->size);                    // Copy from c1-2 fragment (rknn_index: 1)
      memcpy(input_mems[rknn_index][1]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float>(0), input_mems[rknn_index][1]->size);     // Copy from c1-5 fragment (tfl_index: 1)

      // Clear NPU computation time
      ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
      if (ret) {
          perror("ioctl");
          exit(errno);
      }

      // c1-6: Run RKNN inference
      ret = rknn_run(ctx[rknn_index], NULL);
      if (ret < 0) {
        printf("rknn run error %d\n", ret);
        return -1;
      }

      // Get NPU computation time
      ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_GET_DRV_COMP_PERF_NUMS, &drv_comp_perf_eval);
      if (ret) {
          perror("ioctl");
          exit(errno);
      }

      npu_time = npu_time + drv_comp_perf_eval.cnt_total;

      // Run encoder layer 2-5
      for (int i = 0; i < 4; ++i) {
          // c1-2: Copy intermediate results
          rknn_index++;
          memcpy(input_mems[rknn_index][0]->virt_addr, output_mems[rknn_index-1][0]->virt_addr, output_mems[rknn_index-1][0]->size);

          // Clear NPU computation time
          ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
          if (ret) {
              perror("ioctl");
              exit(errno);
          }

          // c1-2: Run RKNN inference
          ret = rknn_run(ctx[rknn_index], NULL);
          if (ret < 0) {
            printf("rknn run error %d\n", ret);
            return -1;
          }

          // Get NPU computation time
          ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_GET_DRV_COMP_PERF_NUMS, &drv_comp_perf_eval);
          if (ret) {
              perror("ioctl");
              exit(errno);
          }

          npu_time = npu_time + drv_comp_perf_eval.cnt_total;

          // c1-3: Copy intermediate results
          for (int i = 0; i < 2; ++i) {
              tfl_input_tensors[i] = tfl_interpreter[tfl_index]->typed_input_tensor<float>(i);
              if (!tfl_input_tensors[i]) {
                  printf("Failed to construct tfl_input_tensor c1-3 %d\n", i);
              }
          }
          memcpy(tfl_input_tensors[0], output_mems[rknn_index][1]->virt_addr, output_mems[rknn_index][1]->size);
          memcpy(tfl_input_tensors[1], output_mems[rknn_index][0]->virt_addr, output_mems[rknn_index][0]->size);

          time_1 = get_timer_counter();

          // c1-3: Run TFLite inference
          tfl_interpreter[tfl_index]->Invoke();

          time_2 = get_timer_counter();
          cpu_time = cpu_time + (time_2 - time_1);

          // c1-4: Copy intermediate results
          rknn_index++;
          memcpy(input_mems[rknn_index][0]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float>(0), input_mems[rknn_index][0]->size);         // Copy from c1-3 fragment
          memcpy(input_mems[rknn_index][1]->virt_addr, output_mems[0][0]->virt_addr, output_mems[0][0]->size);                                              // Copy from c1-1 fragment
          memcpy(input_mems[rknn_index][2]->virt_addr, output_mems[0][1]->virt_addr, output_mems[0][1]->size);                                              // Copy from c1-1 fragment
          // Clear NPU computation time
          ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
          if (ret) {
              perror("ioctl");
              exit(errno);
          }

          // c1-4: Run RKNN inference
          ret = rknn_run(ctx[rknn_index], NULL);
          if (ret < 0) {
            printf("rknn run error %d\n", ret);
            return -1;
          }

          // Get NPU computation time
          ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_GET_DRV_COMP_PERF_NUMS, &drv_comp_perf_eval);
          if (ret) {
              perror("ioctl");
              exit(errno);
          }

          npu_time = npu_time + drv_comp_perf_eval.cnt_total;

          // c1-5: Copy intermediate results
          tfl_index++;
          for (int i = 0; i < 2; ++i) {
              tfl_input_tensors[i] = tfl_interpreter[tfl_index]->typed_input_tensor<float>(i);
              if (!tfl_input_tensors[i]) {
                  printf("Failed to construct tfl_input_tensor %d\n", i);
              }
          }
          memcpy(tfl_input_tensors[0], output_mems[rknn_index-1][2]->virt_addr, output_mems[rknn_index-1][2]->size);     // Copy from c1-2 fragment
          memcpy(tfl_input_tensors[1], output_mems[rknn_index][0]->virt_addr, output_mems[rknn_index][0]->size);         // Copy from c1-4 fragment

          time_1 = get_timer_counter();

          // c1-5: Run TFLite inference
          tfl_interpreter[tfl_index]->Invoke();

          time_2 = get_timer_counter();
          cpu_time = cpu_time + (time_2 - time_1);

          // c1-6: Copy intermediate results
          rknn_index++;
          memcpy(input_mems[rknn_index][0]->virt_addr, output_mems[rknn_index-2][3]->virt_addr, output_mems[rknn_index-2][3]->size);                    // Copy from c1-2 fragment
          memcpy(input_mems[rknn_index][1]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float>(0), input_mems[rknn_index][1]->size);     // Copy from c1-5 fragment

          // Clear NPU computation time
          ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
          if (ret) {
              perror("ioctl");
              exit(errno);
          }

          // c1-6: Run RKNN inference
          ret = rknn_run(ctx[rknn_index], NULL);
          if (ret < 0) {
            printf("rknn run error %d\n", ret);
            return -1;
          }

          // Get NPU computation time
          ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_GET_DRV_COMP_PERF_NUMS, &drv_comp_perf_eval);
          if (ret) {
              perror("ioctl");
              exit(errno);
          }

          npu_time = npu_time + drv_comp_perf_eval.cnt_total;
      }

      // Detach RKNPU
      ioctl_ret = ioctl(ioctl_fd, IOCTL_DETACH_DEVICE);
      if (ioctl_ret) {
        perror("detach device failed");
        exit(errno);
      }

      // Get top 5
      uint32_t topNum = 5;
      uint32_t MaxClass[topNum];
      for (uint32_t i = 0; i < io_num[num_rknn_models-1].n_output; i++) {
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

      // Copy output into shared device
      memcpy(mem + MEM_SIZE, &MaxClass[0], sizeof(uint32_t));
      msync(mem, MEM_SIZE + sizeof(uint32_t), 0);

      // Send message back to vm_service
      // NOTE: The timer counter must be running at 24 MHz
      snprintf(buf, BUF_SIZE, "%.2f,%.2f,%.2f", (npu_time + cpu_time) / 24000.0,
                npu_time / 24000.0, cpu_time / 24000.0);
      send(peer_fd, (void*) buf, BUF_SIZE, 0);
    }
  }

  // Attach RKNPU
  ioctl_ret = ioctl(ioctl_fd, IOCTL_ATTACH_DEVICE);
  if (ioctl_ret) {
    printf("attach device failed\n");
    exit(errno);
  }

  // Clean-ups
  for (int i = 0; i < num_rknn_models; i++) {
    for (uint32_t j = 0; j < io_num[i].n_input; ++j) {
      rknn_destroy_mem(ctx[i], input_mems[i][j]);
    }

    for (uint32_t j = 0; j < io_num[i].n_output; ++j) {
      rknn_destroy_mem(ctx[i], output_mems[i][j]);
    }

    rknn_destroy(ctx[i]);

    free(rknn_model[i]);
    free(input_attrs[i]);
    free(output_attrs[i]);
  }

  // Detach RKNPU
  ioctl_ret = ioctl(ioctl_fd, IOCTL_DETACH_DEVICE);
  if (ioctl_ret) {
    perror("detach device failed");
    exit(errno);
  }

  munmap(mem, MEM_SIZE + 4096);
  close(mem_fd);
  close(ioctl_fd);
  close(vfd);

  return 0;
}

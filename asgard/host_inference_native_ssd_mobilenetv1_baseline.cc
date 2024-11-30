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
#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <drm/drm.h>
#include <linux/types.h>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "include/stb/stb_image_resize.h"

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
  int num_rknn_models = 13;
  int num_tfl_models = 1;
  char* input_path = argv[num_rknn_models + num_tfl_models + 1];

  int loop_count = 1;
  if (argc > (num_rknn_models + num_tfl_models + 2)) {
    loop_count = atoi(argv[num_rknn_models + num_tfl_models + 2]);
  }

  if (argc < (num_rknn_models + num_tfl_models + 2)) {
    printf("Usage:%s rknn_models... tfl_models... input_path [loop_count]\n", argv[0]);
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

  // Bind the current process to cpu7
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(7, &mask);
  ret = sched_setaffinity(0, sizeof(mask), &mask);
  if (ret < 0) {
    perror("sched_setaffinity");
    exit(errno);
  }

  fprintf(output_file, "iteration,inference total,npu time,cpu time\n");

  for (int cnt = 0; cnt < loop_count; cnt++) {
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

    int rknn_index = 0;
    int tfl_index = 0;
    float16_t* tfl_input_tensor = NULL;

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

    // Clear NPU computation time
    ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
    if (ret) {
        perror("ioctl");
        exit(errno);
    }

    // b1: Run RKNN inference
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

    // b2: Copy intermediate results from b1 (rknn_index: 0)
    rknn_index++;
    memcpy(input_mems[rknn_index][0]->virt_addr, output_mems[rknn_index-1][1]->virt_addr, output_mems[rknn_index-1][1]->size);

    // Clear NPU computation time
    ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
    if (ret) {
        perror("ioctl");
        exit(errno);
    }

    // b2: Run RKNN inference
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

    // b3: Copy intermediate results from b1 (rknn_index: 0)
    rknn_index++;
    memcpy(input_mems[rknn_index][0]->virt_addr, output_mems[rknn_index-2][1]->virt_addr, output_mems[rknn_index-2][1]->size);

    // Clear NPU computation time
    ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
    if (ret) {
        perror("ioctl");
        exit(errno);
    }

    // b3: Run RKNN inference
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

    // b4: Copy intermediate results from b3 (rknn_index: 2)
    rknn_index++;
    memcpy(input_mems[rknn_index][0]->virt_addr, output_mems[rknn_index-1][1]->virt_addr, output_mems[rknn_index-1][1]->size);

    // Clear NPU computation time
    ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
    if (ret) {
        perror("ioctl");
        exit(errno);
    }

    // b4: Run RKNN inference
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

    // b5: Copy intermediate results from b3 (rknn_index: 2)
    rknn_index++;
    memcpy(input_mems[rknn_index][0]->virt_addr, output_mems[rknn_index-2][1]->virt_addr, output_mems[rknn_index-2][1]->size);

    // Clear NPU computation time
    ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
    if (ret) {
        perror("ioctl");
        exit(errno);
    }

    // b5: Run RKNN inference
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

    // b6: Copy intermediate results from b5 (rknn_index: 4)
    rknn_index++;
    memcpy(input_mems[rknn_index][0]->virt_addr, output_mems[rknn_index-1][1]->virt_addr, output_mems[rknn_index-1][1]->size);

    // Clear NPU computation time
    ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
    if (ret) {
        perror("ioctl");
        exit(errno);
    }

    // b6: Run RKNN inference
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

    // b7: Copy intermediate results from b5 (rknn_index: 4)
    rknn_index++;
    memcpy(input_mems[rknn_index][0]->virt_addr, output_mems[rknn_index-2][1]->virt_addr, output_mems[rknn_index-2][1]->size);

    // Clear NPU computation time
    ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
    if (ret) {
        perror("ioctl");
        exit(errno);
    }

    // b7: Run RKNN inference
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

    // b8: Copy intermediate results from b7 (rknn_index: 6)
    rknn_index++;
    memcpy(input_mems[rknn_index][0]->virt_addr, output_mems[rknn_index-1][1]->virt_addr, output_mems[rknn_index-1][1]->size);

    // Clear NPU computation time
    ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
    if (ret) {
        perror("ioctl");
        exit(errno);
    }

    // b8: Run RKNN inference
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

    // b9: Copy intermediate results from b7 (rknn_index: 6)
    rknn_index++;
    memcpy(input_mems[rknn_index][0]->virt_addr, output_mems[rknn_index-2][1]->virt_addr, output_mems[rknn_index-2][1]->size);

    // Clear NPU computation time
    ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
    if (ret) {
        perror("ioctl");
        exit(errno);
    }

    // b9: Run RKNN inference
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

    // b10: Copy intermediate results from b9 (rknn_index: 8)
    rknn_index++;
    memcpy(input_mems[rknn_index][0]->virt_addr, output_mems[rknn_index-1][0]->virt_addr, output_mems[rknn_index-1][0]->size);

    // Clear NPU computation time
    ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
    if (ret) {
        perror("ioctl");
        exit(errno);
    }

    // b10: Run RKNN inference
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

    // b11: Copy intermediate results from b9 (rknn_index: 8)
    rknn_index++;
    memcpy(input_mems[rknn_index][0]->virt_addr, output_mems[rknn_index-2][0]->virt_addr, output_mems[rknn_index-2][0]->size);

    // b11: Run RKNN inference
    ret = rknn_run(ctx[rknn_index], NULL);
    if (ret < 0) {
      printf("rknn run error %d\n", ret);
      return -1;
    }

    // b12: Copy intermediate results from b9 (rknn_index: 8)
    rknn_index++;
    memcpy(input_mems[rknn_index][0]->virt_addr, output_mems[rknn_index-3][0]->virt_addr, output_mems[rknn_index-3][0]->size);

    // b12: Run RKNN inference
    ret = rknn_run(ctx[rknn_index], NULL);
    if (ret < 0) {
      printf("rknn run error %d\n", ret);
      return -1;
    }

    // b13: Copy intermediate results
    float16_t* tfl_input_tensors_b13[12];
    for (int i = 0; i < 12; ++i) {
        tfl_input_tensors_b13[i] = tfl_interpreter[tfl_index]->typed_input_tensor<float16_t>(i);
        if (!tfl_input_tensors_b13[i]) {
            printf("Failed to construct tfl_input_tensor %d\n", i);
        }
    }
    memcpy(tfl_input_tensors_b13[0], output_mems[0][0]->virt_addr, output_mems[0][0]->size);      // Copy from b1 fragment (rknn_index: 0)
    memcpy(tfl_input_tensors_b13[1], output_mems[5][0]->virt_addr, output_mems[5][0]->size);      // Copy from b6_2 fragment (rknn_index: 5)
    memcpy(tfl_input_tensors_b13[2], output_mems[10][0]->virt_addr, output_mems[10][0]->size);    // Copy from b11 fragment (rknn_index: 10)
    memcpy(tfl_input_tensors_b13[3], output_mems[11][0]->virt_addr, output_mems[11][0]->size);    // Copy from b12 fragment (rknn_index: 11)
    memcpy(tfl_input_tensors_b13[4], output_mems[3][0]->virt_addr, output_mems[3][0]->size);      // Copy from b4_2 fragment (rknn_index: 3)
    memcpy(tfl_input_tensors_b13[5], output_mems[7][0]->virt_addr, output_mems[7][0]->size);      // Copy from b8_2 fragment (rknn_index: 7)
    memcpy(tfl_input_tensors_b13[6], output_mems[6][0]->virt_addr, output_mems[6][0]->size);      // Copy from b7_2 fragment (rknn_index: 6)
    memcpy(tfl_input_tensors_b13[7], output_mems[2][0]->virt_addr, output_mems[2][0]->size);      // Copy from b3_2 fragment (rknn_index: 2)
    memcpy(tfl_input_tensors_b13[8], output_mems[1][0]->virt_addr, output_mems[1][0]->size);      // Copy from b2_2 fragment (rknn_index: 1)
    memcpy(tfl_input_tensors_b13[9], output_mems[9][0]->virt_addr, output_mems[9][0]->size);      // Copy from b10 fragment (rknn_index: 9)
    memcpy(tfl_input_tensors_b13[10], output_mems[11][0]->virt_addr, output_mems[11][0]->size);   // Copy from b12 fragment (rknn_index: 11)
    memcpy(tfl_input_tensors_b13[11], output_mems[4][0]->virt_addr, output_mems[4][0]->size);     // Copy from b5_2 fragment (rknn_index: 4)

    time_1 = get_timer_counter();

    // b13: Run TFLite inference
    tfl_interpreter[tfl_index]->Invoke();

    time_2 = get_timer_counter();
    cpu_time = cpu_time + (time_2 - time_1);

    // b14: Copy intermediate results
    rknn_index++;
    memcpy(input_mems[rknn_index][0]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float16_t>(9), input_mems[rknn_index][0]->size);
    memcpy(input_mems[rknn_index][1]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float16_t>(10), input_mems[rknn_index][1]->size);
    memcpy(input_mems[rknn_index][2]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float16_t>(8), input_mems[rknn_index][2]->size);
    memcpy(input_mems[rknn_index][3]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float16_t>(11), input_mems[rknn_index][3]->size);
    memcpy(input_mems[rknn_index][4]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float16_t>(6), input_mems[rknn_index][4]->size);
    memcpy(input_mems[rknn_index][5]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float16_t>(7), input_mems[rknn_index][5]->size);
    memcpy(input_mems[rknn_index][6]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float16_t>(2), input_mems[rknn_index][6]->size);
    memcpy(input_mems[rknn_index][7]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float16_t>(4), input_mems[rknn_index][7]->size);
    memcpy(input_mems[rknn_index][8]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float16_t>(5), input_mems[rknn_index][8]->size);
    memcpy(input_mems[rknn_index][9]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float16_t>(0), input_mems[rknn_index][9]->size);
    memcpy(input_mems[rknn_index][10]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float16_t>(1), input_mems[rknn_index][10]->size);
    memcpy(input_mems[rknn_index][11]->virt_addr, tfl_interpreter[tfl_index]->typed_output_tensor<float16_t>(3), input_mems[rknn_index][11]->size);

    // Clear NPU computation time
    ret = ioctl(rknpu_fd, DRM_IOCTL_RKNPU_CLEAR_DRV_COMP_PERF_NUMS);
    if (ret) {
        perror("ioctl");
        exit(errno);
    }

    // b14: Run RKNN inference
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

    // NOTE: The timer counter must be running at 24 MHz
    fprintf(output_file, "%i,%.2f,%.2f,%.2f\n", cnt, (npu_time + cpu_time) / 24000.0,
              npu_time / 24000.0, cpu_time  / 24000.0);
  }

  fclose(output_file);

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

  if (input_data != nullptr) {
    free(input_data);
  }

  return 0;
}

/*-------------------------------------------
                Includes
-------------------------------------------*/
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
#include <linux/vm_sockets.h>

#define BUF_SIZE 64
#define MEM_SIZE 0x4000000

/*-------------------------------------------
                  Functions
-------------------------------------------*/
static inline unsigned long long get_timer_counter(void)
{
  unsigned long long val;
  asm volatile("isb; mrs %0, cntpct_el0" : "=r"(val));
  return val;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char* argv[])
{
  int num_tfl_models = atoi(argv[1]);

  if (argc < (num_tfl_models + 1)) {
    printf("Usage:%s num_tfl_models tfl_models...\n", argv[0]);
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

  // Setup TFLite runtime
  char** tfl_model_path = (char**) malloc(sizeof(char*) * num_tfl_models);
  std::unique_ptr<tflite::FlatBufferModel>* tfl_model = new std::unique_ptr<tflite::FlatBufferModel>[num_tfl_models];
  tflite::ops::builtin::BuiltinOpResolver* tfl_resolver = new tflite::ops::builtin::BuiltinOpResolver[num_tfl_models];
  std::unique_ptr<tflite::Interpreter>* tfl_interpreter = new std::unique_ptr<tflite::Interpreter>[num_tfl_models];

  for (int i = 0; i < num_tfl_models; i++) {
    tfl_model_path[i] = argv[i + 2];

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

  // Open and prepared shared memory device
  int fd = open("/dev/io-mem", O_RDWR | O_SYNC);
  if (fd < 0) {
    perror("open");
    exit(errno);
  }

  char* mem = (char*) mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
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

  socklen_t peer_addr_size = sizeof(struct sockaddr_vm);
  while (1) {
    struct sockaddr_vm peer_addr;
    int peer_fd = accept(s, (sockaddr*) &peer_addr, &peer_addr_size);
    if (peer_fd < 0) {
      perror("accept");
      exit(errno);
    }

    size_t msg_len;
    int tfl_index = 0;
    while ((msg_len = recv(peer_fd, buf, BUF_SIZE, 0)) > 0) {
      unsigned long long vsock_send_end_time = get_timer_counter();

      // Copy intermediate results
      float* tfl_input_tensor_1 = tfl_interpreter[tfl_index]->typed_input_tensor<float>(0);
      if (!tfl_input_tensor_1){
        printf("Failed to construct tfl_input_tensor\n");
      }
      TfLiteIntArray* tfl_input_dims_1 = tfl_interpreter[tfl_index]->input_tensor(0)->dims;
      size_t tfl_input_size_1 = sizeof(float);
      for (int i = 0; i < tfl_input_dims_1->size; i++) {
          tfl_input_size_1 *= tfl_input_dims_1->data[i];
      }
      memcpy(tfl_input_tensor_1, mem, tfl_input_size_1);

      // Run (Transpose)
      tfl_interpreter[tfl_index]->Invoke();

      // Copy intermediate results
      int tfl_output_1 = tfl_interpreter[tfl_index]->outputs()[0];
      TfLiteIntArray* tfl_output_dims_1 = tfl_interpreter[tfl_index]->tensor(tfl_output_1)->dims;
      auto tfl_output_size_1 = sizeof(float);
      for (int i = 0; i < tfl_output_dims_1->size; ++i) {
          tfl_output_size_1 *= tfl_output_dims_1->data[i];
      }
      float* tfl_input_tensor_2 = tfl_interpreter[tfl_index+1]->typed_input_tensor<float>(0);
      if (!tfl_input_tensor_2){
        printf("Failed to construct tfl_input_tensor\n");
      }
      memcpy(tfl_input_tensor_2, tfl_interpreter[tfl_index]->typed_output_tensor<float>(0), tfl_output_size_1);

      unsigned long long deobfuscate_time = get_timer_counter();

      // Run (Add, Mul, and Sub)
      tfl_interpreter[tfl_index+1]->Invoke();

      deobfuscate_time = get_timer_counter() - deobfuscate_time;

      float* tfl_input_tensor_3 = tfl_interpreter[tfl_index+2]->typed_input_tensor<float>(0);
      if (!tfl_input_tensor_3){
        printf("Failed to construct tfl_input_tensor\n");
      }
      memcpy(tfl_input_tensor_3, tfl_interpreter[tfl_index]->typed_output_tensor<float>(0), tfl_output_size_1);

      unsigned long long relu_time = get_timer_counter();

      // Run (ReLU)
      tfl_interpreter[tfl_index+2]->Invoke();

      relu_time = get_timer_counter() - relu_time;

      // Copy intermediate results
      int tfl_output_3 = tfl_interpreter[tfl_index+2]->outputs()[0];
      TfLiteIntArray* tfl_output_dims_3 = tfl_interpreter[tfl_index+2]->tensor(tfl_output_3)->dims;
      auto tfl_output_size_3 = sizeof(float);
      for (int i = 0; i < tfl_output_dims_3->size; ++i) {
          tfl_output_size_3 *= tfl_output_dims_3->data[i];
      }
      float* tfl_input_tensor_4 = tfl_interpreter[tfl_index+3]->typed_input_tensor<float>(0);
      if (!tfl_input_tensor_4){
        printf("Failed to construct tfl_input_tensor\n");
      }
      memcpy(tfl_input_tensor_4, tfl_interpreter[tfl_index+2]->typed_output_tensor<float>(0), tfl_output_size_3);

      unsigned long long obfuscate_time = get_timer_counter();

      // Run (Add)
      tfl_interpreter[tfl_index+3]->Invoke();

      obfuscate_time = get_timer_counter() - obfuscate_time;

      float* tfl_input_tensor_5 = tfl_interpreter[tfl_index+4]->typed_input_tensor<float>(0);
      if (!tfl_input_tensor_5){
        printf("Failed to construct tfl_input_tensor\n");
      }
      memcpy(tfl_input_tensor_5, tfl_interpreter[tfl_index+2]->typed_output_tensor<float>(0), tfl_output_size_3);

      // Run (Transpose)
      tfl_interpreter[tfl_index+4]->Invoke();

      // Copy intermediate results
      int tfl_output_5 = tfl_interpreter[tfl_index+4]->outputs()[0];
      TfLiteIntArray* tfl_output_dims_5 = tfl_interpreter[tfl_index+4]->tensor(tfl_output_5)->dims;
      auto tfl_output_size_5 = sizeof(float);
      for (int i = 0; i < tfl_output_dims_5->size; ++i) {
          tfl_output_size_5 *= tfl_output_dims_5->data[i];
      }
      memcpy(mem, tfl_interpreter[tfl_index+4]->typed_output_tensor<float>(0), tfl_output_size_5);

      ret = msync(mem, tfl_output_size_5, 0);
      if (ret < 0) {
        perror("msync");
        exit(errno);
      }

      // Weird, but VSOCK made me do this
      tfl_index = tfl_index + 5;
      if (tfl_index >= num_tfl_models) {
          tfl_index = 0;
      }

      unsigned long long vsock_recv_start_time = get_timer_counter();

      // Send message back to vm_service
      snprintf(buf, BUF_SIZE, "%llu,%llu,%llu,%llu,%llu",
               vsock_send_end_time, deobfuscate_time, relu_time, obfuscate_time, vsock_recv_start_time);
      send(peer_fd, (void*) buf, BUF_SIZE, 0);
    }
  }

  munmap(mem, MEM_SIZE);
  close(fd);
  close(vfd);

  return 0;
}

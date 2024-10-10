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
#include <sys/resource.h>

#define BUF_SIZE 64
#define MEM_SIZE 224 * 224 * 3

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char* argv[])
{
  if (argc < 3) {
    printf("Usage:%s socket_path shared_memory_path [loop_count]\n", argv[0]);
    return -1;
  }

  char* sock_path = argv[1];
  char* mem_path = argv[2];

  int loop_count = 1;
  if (argc > 3) {
    loop_count = atoi(argv[3]);
  }

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

  int ret = connect(sfd, (struct sockaddr *) &my_addr, sizeof(my_addr));
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


  fprintf(output_file, "iteration,inference only,inference and hypercall\n");

  char* buf = (char*) malloc(BUF_SIZE);
  if (buf == NULL) {
      perror("malloc");
      exit(errno);
  }

  // Do not measure time for open() and mmap()
  int fd = open(mem_path, O_RDWR | O_SYNC);
  if (fd < 0) {
    perror("open");
    exit(errno);
  }

  char* mem = (char*) mmap(NULL, MEM_SIZE + 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mem == MAP_FAILED) {
    perror("mmap");
    exit(errno);
  }

  ret = msync(mem, MEM_SIZE, 0);
  if (ret < 0) {
    perror("msync");
    exit(errno);
  }

  printf("Begin running %i inferences\n", loop_count);
  for (int cnt = 0; cnt < loop_count; cnt++) {
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

    // Output timer measurments to output.csv
    fprintf(output_file, "%i,", cnt);
    fprintf(output_file, "%.60s\n", buf);
    // Print inference result
    uint32_t output[1];
    memcpy(output, mem + MEM_SIZE, sizeof(uint32_t));
    memset(mem + MEM_SIZE, 0, sizeof(uint32_t));
    printf("Result: %u\n", *output);
    printf("Iteration: %d/%d\n", cnt + 1, loop_count);
  }

  munmap(mem, MEM_SIZE + 4096);

  close(fd);
  close(sfd);
  fclose(output_file);

  return 0;
}

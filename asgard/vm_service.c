#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdbool.h>
#include <sys/un.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/resource.h>
#include <linux/vm_sockets.h>

#define LISTEN_BACKLOG 50
#define BUF_SIZE 64

int main(int argc, char* argv[])
{
	struct sockaddr_un my_addr, peer_addr;
	socklen_t peer_addr_size;
	char* sock_path;
	int sfd, cfd, cid, ret, s;
	bool listening = true;

	if (argc < 3) {
		printf("Usage:%s cid socket_path\n", argv[0]);
		return -1;
	}

	cid = atoi(argv[1]);
	sock_path = argv[2];

	// Set nice value to the highest priority (-20)
	errno = 0;
	setpriority(PRIO_PROCESS, 0, -20);
	if (errno) {
		perror("setpriority");
		exit(errno);
	}

	// Bind the current process to cpu6
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(6, &mask);
	ret = sched_setaffinity(0, sizeof(mask), &mask);
	if (ret < 0) {
		perror("sched_setaffinity");
		exit(errno);
	}

	// Set up socket server: host app <-> vm service
	sfd = socket(PF_LOCAL, SOCK_STREAM, 0);
	if (sfd == -1) {
		perror("socket");
		exit(errno);
	}

	memset(&my_addr, 0, sizeof(my_addr));
	my_addr.sun_family = AF_LOCAL;
	strncpy(my_addr.sun_path, sock_path, sizeof(my_addr.sun_path) - 1);

	if (bind(sfd, (struct sockaddr *) &my_addr, sizeof(my_addr)) == -1) {
		perror("bind");
		exit(errno);
	}

	if (listen(sfd, LISTEN_BACKLOG) == -1) {
		perror("listen");
		exit(errno);
	}

	printf("vm_service: socket setup complete (host app <-> vm service)\n");

	// Set up vsock client: vm service <-> guest
	s = socket(AF_VSOCK, SOCK_STREAM, 0);
	if (s < 0) {
		printf("vm_service: socket creation failed\n");
		perror("socket");
		exit(errno);
	}

	struct sockaddr_vm addr;
	memset(&addr, 0, sizeof(struct sockaddr_vm));
	addr.svm_family = AF_VSOCK;
	addr.svm_port = 9999;
	addr.svm_cid = cid;

	// Connecting to vsock
	ret = connect(s, &addr, sizeof(struct sockaddr_vm));
	if (ret < 0) {
		perror("connect");
		exit(errno);
	}

	printf("vm_service: vsock setup complete (vm service <-> guest app)\n");

	// Now we can accept incoming connections one at a time using accept(2)
	while (listening) {
		peer_addr_size = sizeof(peer_addr);
		cfd = accept(sfd, (struct sockaddr *)&peer_addr, &peer_addr_size);
		if (cfd == -1) {
			perror("accept");
			exit(errno);
		}

		char *buf = (char*)malloc(BUF_SIZE);
		size_t msg_len;
		while ((msg_len = recv(cfd, buf, BUF_SIZE, 0)) > 0) {
			// Relay a request to guest VM

			// Sending request to guest app
			ret = send(s, "start inference", 15, 0);
			if (ret < 0) {
				printf("vm_service: send detected local error\n");
				perror("send");
				exit(errno);
			}

			// Wait for guest app to reply
			msg_len = recv(s, buf, BUF_SIZE, 0);

			// Reply back to host app
			ret = send(cfd, (void *)buf, BUF_SIZE, 0);
			if (ret < 0) {
				printf("vm_service: send detected local error\n");
				perror("send");
				exit(errno);
			}
		}
	}

	// When no longer required, the socket should be deleted
	// using unlink(2) or remove(3)
	close(sfd);
	unlink(sock_path);

	close(s);

	return 0;
}

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>
#include <string>
#include <thread>

#include "rb10_rmpflow_rviz/rb10_socket_client.hpp"

namespace
{

constexpr int kCommandPort = 5000;
constexpr int kDataPort = 5001;

int open_loopback_listener(int port)
{
  const int socket_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd < 0) {
    return -1;
  }

  const int enabled = 1;
  ::setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<uint16_t>(port));
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (
    ::bind(socket_fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
    ::listen(socket_fd, 1) != 0)
  {
    ::close(socket_fd);
    return -1;
  }
  return socket_fd;
}

TEST(Rb10SocketClientReadOnly, OpensOnlyDataChannelAndRejectsCommands)
{
  const int command_listener = open_loopback_listener(kCommandPort);
  const int data_listener = open_loopback_listener(kDataPort);
  ASSERT_GE(command_listener, 0) << std::strerror(errno);
  ASSERT_GE(data_listener, 0) << std::strerror(errno);

  std::promise<std::string> data_request_promise;
  auto data_request_future = data_request_promise.get_future();
  std::mutex release_mutex;
  std::condition_variable release_cv;
  bool release_data_peer = false;

  std::thread data_server([&]() {
    pollfd listener_poll{data_listener, POLLIN, 0};
    if (::poll(&listener_poll, 1, 2000) <= 0) {
      data_request_promise.set_value("accept_timeout");
      return;
    }

    const int peer = ::accept(data_listener, nullptr, nullptr);
    if (peer < 0) {
      data_request_promise.set_value("accept_failed");
      return;
    }

    char request[7]{};
    std::size_t received_total = 0U;
    while (received_total < sizeof(request)) {
      const ssize_t received =
        ::recv(peer, request + received_total, sizeof(request) - received_total, 0);
      if (received <= 0) {
        break;
      }
      received_total += static_cast<std::size_t>(received);
    }
    data_request_promise.set_value(std::string(request, received_total));

    std::unique_lock<std::mutex> lock(release_mutex);
    release_cv.wait_for(lock, std::chrono::seconds(2), [&]() {return release_data_peer;});
    ::close(peer);
  });

  rb10_rmpflow_rviz::Rb10SocketClient client;
  const bool connected = client.connect(
    "127.0.0.1",
    20.0,
    [](const rb10_rmpflow_rviz::Rb10SystemState &) {},
    [](const std::string &) {},
    false,
    1,
    false);
  EXPECT_TRUE(connected);

  const auto request_status = data_request_future.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(request_status, std::future_status::ready);
  if (request_status == std::future_status::ready) {
    EXPECT_EQ(data_request_future.get(), "reqdata");
  }
  EXPECT_FALSE(client.send_text_command("move_speed_j(jnt[0,0,0,0,0,0],0.02,0.2,0.05,0.1)"));

  pollfd command_poll{command_listener, POLLIN, 0};
  EXPECT_EQ(::poll(&command_poll, 1, 250), 0)
    << "read-only mode unexpectedly connected to the RB10 command port";

  client.disconnect();
  {
    std::scoped_lock lock(release_mutex);
    release_data_peer = true;
  }
  release_cv.notify_all();
  data_server.join();
  ::close(command_listener);
  ::close(data_listener);
}

}  // namespace

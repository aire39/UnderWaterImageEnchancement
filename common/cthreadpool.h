#pragma once

#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "cjthread.h"
#include "cthread.h"

class cthreadpool
{
  public:
    cthreadpool() = delete;
    explicit cthreadpool(size_t n_threads);
    explicit cthreadpool(const size_t && n_threads, const std::string && t_pool_name);
    explicit cthreadpool(size_t n_threads, const std::string & t_pool_name);
    ~cthreadpool();

    void addjob(const std::function<void()>& job);
    void addjob(std::function<void()>&& job) noexcept;
    void waitforthread();
    void ForceCancelThreadWait();

    [[nodiscard]] size_t threadswaiting();
    [[nodiscard]] size_t threadsinuse();
    [[nodiscard]] size_t numberofthreads() const;
    [[nodiscard]] size_t numberofjobs() const;

  private:
    void threadpooltask(size_t thread_index);

    std::vector<cjthread> threads;
    std::vector<bool> threadInUse;
    std::queue<std::function<void()>> queuedJobs;
    std::recursive_mutex qrmutex;
    std::mutex jobmutex;
    std::mutex checkmutex;
    std::condition_variable cvJobAvailable;
    std::condition_variable cvCheckForFreeThread;
    std::string name = "tp";
    size_t nThreads = 0;
    bool running = true;
    bool forceCancelWait = false;
};

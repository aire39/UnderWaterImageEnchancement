#pragma once

#include <thread>
#include <string>
#include <functional>

class cjthread
{
  public:
    cjthread()
    {
    }

    template<typename callable, typename... argslist>
    explicit cjthread(callable&& func, argslist&&... args)
      : thread
      (&cjthread::cinit
      ,this
      ,std::forward<std::string>(threadName)
      ,std::forward<std::string>("No description")
      ,std::bind(std::forward<callable>(func) ,std::forward<argslist>(args)...)
      )
    {
    }

    template<typename callable, typename... argslist>
    explicit cjthread(std::string && thread_name, callable&& func, argslist&&... args)
      : thread
      (&cjthread::cinit
      ,this
      ,std::forward<std::string>(thread_name)
      ,std::forward<std::string>("No description")
      ,std::bind(std::forward<callable>(func), std::forward<argslist>(args)...)
      )
    {
    }

    template<typename callable, typename... argslist>
    explicit cjthread(std::string thread_name, callable&& func, argslist&&... args)
      : thread
      (&cjthread::cinit
      ,this
      ,std::forward<std::string>(thread_name)
      ,std::forward<std::string>("No description")
      ,std::bind(std::forward<callable>(func), std::forward<argslist>(args)...)
      )
    {
    }

    template<typename callable, typename... argslist>
    explicit cjthread(const char * thread_name, callable&& func, argslist&&... args)
      : thread
      (&cjthread::cinit
      ,this
      ,std::forward<std::string>(thread_name)
      ,std::forward<std::string>("No description")
      ,std::bind(std::forward<callable>(func), std::forward<argslist>(args)...)
      )
    {
    }

    template<typename callable, typename... argslist>
    explicit cjthread(std::string && thread_name, std::string && thread_description, callable&& func, argslist&&... args)
      : thread
      (&cjthread::cinit
      ,this
      ,std::forward<std::string>(thread_name)
      ,std::forward<std::string>(thread_description)
      ,std::bind(std::forward<callable>(func), std::forward<argslist>(args)...)
      )
    {
    }

    template<typename callable, typename... argslist>
    explicit cjthread(const char * thread_name, const char * thread_description, callable&& func, argslist&&... args)
      : thread
      (&cjthread::cinit
      ,this
      ,std::forward<std::string>(thread_name)
      ,std::forward<std::string>(thread_description)
      ,std::bind(std::forward<callable>(func), std::forward<argslist>(args)...)
      )
    {
    }

    void setname(std::string name);
    void setdescription(std::string description);

    std::string name();
    std::string description();

  private:
    std::jthread thread;

    void cinit(std::string && name, std::string && description, std::function<void()> && task)
    {
      threadName = name;
      threadDesc = description;

      setname(name);

      task();
    }

    std::string threadName = "Unnamed";
    std::string threadDesc = "No description";
};

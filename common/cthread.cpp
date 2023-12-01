#include "cthread.h"

#if defined(_WIN32) || defined(WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <climits>
#include <processthreadsapi.h>

#include <pshpack8.h>
typedef struct {
  DWORD dwType;
  LPCSTR szName;
  DWORD dwThreadID;
  DWORD dwFlags;
} THREADNAME_INFO;
#include <poppack.h>

static EXCEPTION_DISPOSITION NTAPI ignore_handler(EXCEPTION_RECORD *rec,
                                                  void *frame, CONTEXT *ctx,
                                                  void *disp)
{
  return ExceptionContinueExecution;
}

#if (defined(__MINGW32__) || defined(__MINGW64__))
#include <pthread.h>
#include <winnt.h>
#include <winternl.h>
#endif
#elif defined(linux) || defined(unix)
#include <pthread.h>
#endif

#if defined(_WIN32) || defined(WIN32)
// set thread name for windows
//https://gist.github.com/rossy/7faf0ab90a54d6b5a46f
void cthread::setname(std::string name)
{

  static const DWORD MS_VC_EXCEPTION = 0x406D1388;

  // Don't bother if a debugger isn't attached to receive the event
  if (!IsDebuggerPresent())
    return;

  // Thread information for VS compatible debugger. -1 sets current thread.
  THREADNAME_INFO ti = {
      .dwType = 0x1000,
      .szName = name.c_str(),
      .dwThreadID = std::numeric_limits<DWORD>::max(),
  };

  // Push an exception handler to ignore all following exceptions
  NT_TIB *tib = ((NT_TIB*)NtCurrentTeb());
  EXCEPTION_REGISTRATION_RECORD rec = {
      .Next = tib->ExceptionList,
      .Handler = ignore_handler,
  };
  tib->ExceptionList = &rec;

  // Visual Studio and compatible debuggers receive thread names from the
  // program through a specially crafted exception
  RaiseException(MS_VC_EXCEPTION, 0, sizeof(ti) / sizeof(ULONG_PTR),
                 (ULONG_PTR*)&ti);

  // Pop exception handler
  tib->ExceptionList = tib->ExceptionList->Next;
}
#elif defined(linux) || defined(unix)
void cthread::setname(std::string name)
{
  if (name.size() > 15)
  {
    name[15] = '\0';
  }

  pthread_setname_np(native_handle(), name.c_str());
}
#endif

std::string cthread::name()
{
  return threadName;
}

std::string cthread::description()
{
  return threadDesc;
}

void cthread::cinit(std::string && name, std::string && description, std::function<void()> && task) noexcept
{
  threadName = name;
  threadDesc = description;

  setname(name);

  task();
}
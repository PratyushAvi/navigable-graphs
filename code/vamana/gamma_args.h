#pragma once

// Reads -gamma and -S from the process's own command line.
//
// These flags belong to this project's Vamana variant, but the driver that
// parses the command line (ParlayANN's bench/neighborsTime.C) and the struct it
// fills (utils/types.h) are upstream files. Putting the flags there means every
// machine needs those edits applied before the build works -- which is exactly
// how an HPC checkout ends up failing with "BuildParams has no member gamma".
//
// So we read argv ourselves instead, and the ParlayANN checkout stays stock.
// Three ways to get it, tried in order:
//   1. a constructor-attribute function, which GCC and Clang hand (argc, argv)
//      on both glibc and macOS;
//   2. /proc/self/cmdline on Linux;
//   3. _NSGetArgv on macOS.
// If all three fail the flags simply read as absent, so the build still runs as
// stock Vamana rather than misbehaving.

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <crt_externs.h>
#endif

namespace parlayANN {
namespace gamma_args {

inline std::vector<std::string>& captured() {
  static std::vector<std::string> args;
  return args;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
inline void capture_main_args(int argc, char** argv, char**) {
  auto& a = captured();
  if (!a.empty()) return;
  for (int i = 0; i < argc; i++) if (argv[i]) a.push_back(argv[i]);
}
#endif

// Split a NUL-separated /proc/self/cmdline blob into arguments.
inline std::vector<std::string> from_proc_cmdline() {
  std::vector<std::string> out;
  std::ifstream f("/proc/self/cmdline", std::ios::binary);
  if (!f) return out;
  std::string all((std::istreambuf_iterator<char>(f)),
                  std::istreambuf_iterator<char>());
  size_t i = 0;
  while (i < all.size()) {
    size_t j = all.find('\0', i);
    if (j == std::string::npos) j = all.size();
    if (j > i) out.push_back(all.substr(i, j - i));
    i = j + 1;
  }
  return out;
}

inline const std::vector<std::string>& args() {
  static std::vector<std::string> resolved = [] {
    if (!captured().empty()) return captured();
    auto p = from_proc_cmdline();
    if (!p.empty()) return p;
#if defined(__APPLE__)
    std::vector<std::string> m;
    int argc = *_NSGetArgc();
    char** argv = *_NSGetArgv();
    if (argv) for (int i = 0; i < argc; i++) if (argv[i]) m.push_back(argv[i]);
    return m;
#else
    return std::vector<std::string>();
#endif
  }();
  return resolved;
}

// Value following `flag`, or nullptr when the flag is absent or has no value.
inline const std::string* value_of(const std::string& flag) {
  const auto& a = args();
  for (size_t i = 1; i + 1 < a.size(); i++)
    if (a[i] == flag) return &a[i + 1];
  return nullptr;
}

inline double gamma_value() {
  const std::string* v = value_of("-gamma");
  if (!v) return 0.0;
  double g = std::atof(v->c_str());
  if (g < 0.0 || g > 1.0) {
    std::cout << "-gamma must be in (0, 1], got " << *v << std::endl;
    abort();
  }
  return g;
}

inline long sample_size_value() {
  const std::string* v = value_of("-S");
  return v ? std::atol(v->c_str()) : 0;
}

}  // namespace gamma_args
}  // namespace parlayANN

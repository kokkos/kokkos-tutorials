// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  int min_exponent = 7;
  int max_exponent = 27;
  int samples      = 20;
  int warmups      = 3;
  bool use_host_execution_space = false;
  std::string label;
};

volatile double benchmark_sink = 0.0;

void print_usage(char const* program) {
  std::cerr << "Usage: " << program
            << " --label <plot label> [--min-exponent 7]"
               " [--max-exponent 27] [--samples 20] [--warmups 3]"
               " [--host-execution-space]\n";
}

int parse_int(char const* option, char const* value) {
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (*value == '\0' || *end != '\0' || parsed < 0) {
    throw std::runtime_error(std::string("invalid value for ") + option +
                             ": " + value);
  }
  return static_cast<int>(parsed);
}

Options parse_options(int argc, char* argv[]) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string const argument = argv[i];
    auto require_value = [&](char const* option) {
      if (++i == argc) {
        throw std::runtime_error(std::string("missing value for ") + option);
      }
      return argv[i];
    };

    if (argument == "--label") {
      options.label = require_value("--label");
    } else if (argument == "--min-exponent") {
      options.min_exponent =
          parse_int("--min-exponent", require_value("--min-exponent"));
    } else if (argument == "--max-exponent") {
      options.max_exponent =
          parse_int("--max-exponent", require_value("--max-exponent"));
    } else if (argument == "--samples") {
      options.samples = parse_int("--samples", require_value("--samples"));
    } else if (argument == "--warmups") {
      options.warmups = parse_int("--warmups", require_value("--warmups"));
    } else if (argument == "--host-execution-space") {
      options.use_host_execution_space = true;
    } else if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }

  if (options.label.empty()) {
    throw std::runtime_error("--label is required");
  }
  if (options.label.find_first_of(",\r\n") != std::string::npos) {
    throw std::runtime_error("--label must not contain commas or newlines");
  }
  if (options.min_exponent > options.max_exponent) {
    throw std::runtime_error("--min-exponent must not exceed --max-exponent");
  }
  if (options.max_exponent > 62) {
    throw std::runtime_error("--max-exponent must be at most 62");
  }
  if (options.samples == 0) {
    throw std::runtime_error("--samples must be positive");
  }
  return options;
}

KOKKOS_INLINE_FUNCTION double integrand(double x) {
  return 4.0 / (1.0 + x * x);
}

double serial_integral(int64_t intervals) {
  double const dx = 1.0 / static_cast<double>(intervals);
  double result   = 0.0;
  for (int64_t i = 0; i < intervals; ++i) {
    double const x = (static_cast<double>(i) + 0.5) * dx;
    result += integrand(x);
  }
  return result * dx;
}

template <class ExecutionSpace>
double parallel_integral(int64_t intervals) {
  using policy_type =
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<int64_t>>;

  double const dx = 1.0 / static_cast<double>(intervals);
  double result   = 0.0;
  Kokkos::parallel_reduce(
      "scalar_integration", policy_type(0, intervals),
      KOKKOS_LAMBDA(int64_t i, double& update) {
        double const x = (static_cast<double>(i) + 0.5) * dx;
        update += integrand(x);
      },
      result);
  ExecutionSpace().fence();
  return result * dx;
}

template <class Function>
double measure_median(Function&& function, int samples, double& result) {
  std::vector<double> timings;
  timings.reserve(samples);
  for (int sample = 0; sample < samples; ++sample) {
    Kokkos::Timer timer;
    result = function();
    benchmark_sink = result;
    timings.push_back(timer.seconds());
  }
  std::sort(timings.begin(), timings.end());
  std::size_t const middle = timings.size() / 2;
  return timings.size() % 2 == 0
             ? 0.5 * (timings[middle - 1] + timings[middle])
             : timings[middle];
}

}  // namespace

int main(int argc, char* argv[]) {
  Options options;
  try {
    options = parse_options(argc, argv);
  } catch (std::exception const& error) {
    std::cerr << "error: " << error.what() << "\n";
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  Kokkos::initialize(argc, argv);
  {
    std::cerr << "label: " << options.label << "\n";

    std::cout << "label,intervals,serial_seconds,parallel_seconds,speedup,"
                 "serial_result,parallel_result\n";
    std::cout << std::setprecision(17);

    for (int exponent = options.min_exponent;
         exponent <= options.max_exponent; ++exponent) {
      int64_t const intervals = int64_t{1} << exponent;

      double parallel_result = 0.0;
      auto parallel_operation = [&] {
        if (options.use_host_execution_space) {
          return parallel_integral<Kokkos::DefaultHostExecutionSpace>(
              intervals);
        }
        return parallel_integral<Kokkos::DefaultExecutionSpace>(intervals);
      };
      for (int warmup = 0; warmup < options.warmups; ++warmup) {
        parallel_result = parallel_operation();
      }

      double serial_result = 0.0;
      double const serial_seconds = measure_median(
          [&] { return serial_integral(intervals); }, options.samples,
          serial_result);
      double const parallel_seconds = measure_median(
          parallel_operation, options.samples, parallel_result);

      double const tolerance = 1.0e-10 * std::max(1.0, std::abs(serial_result));
      if (std::abs(serial_result - parallel_result) > tolerance) {
        std::cerr << "error: result mismatch at " << intervals
                  << " intervals: serial=" << serial_result
                  << ", parallel=" << parallel_result << "\n";
        Kokkos::finalize();
        return EXIT_FAILURE;
      }

      std::cout << options.label << ',' << intervals << ',' << serial_seconds
                << ',' << parallel_seconds << ','
                << serial_seconds / parallel_seconds << ',' << serial_result
                << ',' << parallel_result << '\n';
    }
  }
  Kokkos::finalize();
  return EXIT_SUCCESS;
}

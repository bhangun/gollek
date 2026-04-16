/**
 * tests/smoke_test_main.cpp
 * Entry point for the standalone smoke-test binary.
 */
#include <cstdio>
#include <cstdlib>

// Declared in platform/desktop/gollek_desktop_runner.cpp
int gollek_run_smoke_test(const char *model_path);

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: gollek_test <model.litertlm>\n");
    return 1;
  }
  return gollek_run_smoke_test(argv[1]);
}

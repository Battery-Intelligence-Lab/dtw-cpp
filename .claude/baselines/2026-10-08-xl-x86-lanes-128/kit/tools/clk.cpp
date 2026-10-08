// Which dependent chain measures the core clock on this CPU? (add-immediate chains can be folded at rename)
#define NOMINMAX
#include <windows.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
static double now() { return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__attribute__((noinline)) void add_imm(std::uint64_t n) { std::uint64_t a = 0; asm volatile("1:\n\t.rept 100\n\taddq $1, %[a]\n\t.endr\n\tdecq %[n]\n\tjnz 1b" : [n] "+r"(n), [a] "+r"(a) : : "cc"); }
__attribute__((noinline)) void add_reg(std::uint64_t n, std::uint64_t b) { std::uint64_t a = 0; asm volatile("1:\n\t.rept 100\n\taddq %[b], %[a]\n\t.endr\n\tdecq %[n]\n\tjnz 1b" : [n] "+r"(n), [a] "+r"(a) : [b] "r"(b) : "cc"); }
__attribute__((noinline)) void imul_chain(std::uint64_t n, std::uint64_t b) { std::uint64_t a = 1; asm volatile("1:\n\t.rept 100\n\timulq %[b], %[a]\n\t.endr\n\tdecq %[n]\n\tjnz 1b" : [n] "+r"(n), [a] "+r"(a) : [b] "r"(b) : "cc"); }
int main(int argc, char **argv) {
  SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << 12);
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
  const std::uint64_t it = 300000, b = argc > 5 ? 3 : 1;
  for (int r = 0; r < 5; ++r) {
    double t0 = now(); add_imm(it); double t1 = now(); add_reg(it, b); double t2 = now(); imul_chain(it, b); double t3 = now();
    std::printf("add-imm %.3f GHz  add-reg %.3f GHz  imul(3 cycles) %.3f GHz\n", it * 100 / (t1 - t0), it * 100 / (t2 - t1), it * 100 * 3 / (t3 - t2));
  }
}

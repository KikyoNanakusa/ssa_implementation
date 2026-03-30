.PHONY: all clean all-static

CC := riscv64-linux-gnu-gcc
CFLAGS := -O0
CFLAGS_STATIC := $(CFLAGS) -static

SRCS := $(wildcard *.c)
BINS := $(patsubst %.c,%.out,$(SRCS))
BINS_STATIC := $(patsubst %.c,%-static.out,$(SRCS))

all: $(BINS)
all-static: $(BINS_STATIC)

%.out: %.c
	$(CC) $(CFLAGS) -o $@ $<

%-static.out: %.c
	$(CC) $(CFLAGS_STATIC) -o $@ $<

clean:
	rm -f $(BINS) $(BINS_STATIC)
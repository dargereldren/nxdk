PBKIT_DIR := $(NXDK_DIR)/lib/pbkit
PBKIT_SRC_DIR := $(PBKIT_DIR)/src
PBKIT_BUILD_DIR := $(PBKIT_DIR)/makebuild
PBKIT_LIB := $(PBKIT_BUILD_DIR)/libpbkit.lib
PBKIT_INSTALLED_LIB := $(NXDK_DIR)/lib/libpbkit.lib

PBKIT_SRCS := \
	$(PBKIT_SRC_DIR)/pbkit.c \
	$(PBKIT_SRC_DIR)/pbkit_gamma.c \
	$(PBKIT_SRC_DIR)/pbkit_dma.c \
	$(PBKIT_SRC_DIR)/pbkit_draw.c \
	$(PBKIT_SRC_DIR)/pbkit_print.c \
	$(PBKIT_SRC_DIR)/pbkit_pushbuffer.c

PBKIT_OBJS := $(patsubst $(PBKIT_SRC_DIR)/%.c,$(PBKIT_BUILD_DIR)/%.obj,$(PBKIT_SRCS))

NXDK_CFLAGS += -I$(PBKIT_DIR)
NXDK_CXXFLAGS += -I$(PBKIT_DIR)

$(PBKIT_BUILD_DIR):
	@mkdir -p $(PBKIT_BUILD_DIR)

$(PBKIT_BUILD_DIR)/%.obj: $(PBKIT_SRC_DIR)/%.c | $(PBKIT_BUILD_DIR)
	@echo "[ CC       ] $@"
	$(VE)$(CC) $(NXDK_CFLAGS) $(CFLAGS) -MD -MP -MT '$@' -MF '$(PBKIT_BUILD_DIR)/$*.c.d' -c -o '$@' '$<'

$(PBKIT_LIB): $(PBKIT_OBJS)
	@echo "[ LIB      ] $@"
	$(VE)$(LIB) -out:'$@' $^

$(PBKIT_INSTALLED_LIB): $(PBKIT_LIB)
	@echo "[ COPY     ] $@"
	$(VE)cp '$<' '$@'

main.exe: $(PBKIT_INSTALLED_LIB)

CLEANRULES += clean-pbkit

.PHONY: clean-pbkit
clean-pbkit:
	$(VE)rm -rf $(PBKIT_BUILD_DIR) $(PBKIT_INSTALLED_LIB)

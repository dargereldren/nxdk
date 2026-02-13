#ifndef _PBGL_PBGL_H
#define _PBGL_PBGL_H

#ifdef __cplusplus
extern "C" {
#endif

#define PBGL_MAXRAM 0x03FFAFFF
#define PBGL_ALLOCFLAGS 0x404

int pbgl_init(int init_pbkit);
void pbgl_swap_buffers(void);
void *pbgl_alloc(unsigned int size);
void pbgl_free(void *addr);
void pbgl_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif // _PBGL_PBGL_H

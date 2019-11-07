CC=gcc
LDLIBS=-lOpenCL
CEXT=.c
OEXT=.out

TARGETS= vecsum$(CEXT) vecsum4$(CEXT) vecsum8$(CEXT) vecsum4by4$(CEXT) vecsum4by2$(CEXT) vecsmooth$(CEXT)
TARGETS+= vecsmooth_lmem$(CEXT) vecsmooth_v4$(CEXT) 
TARGETS+= matinit$(CEXT) transpose$(CEXT) transpose_lmem$(CEXT)

all: $(TARGETS)
	$(CC) vecsum$(CEXT) $(LDLIBS) -o vecsum$(OEXT)
	$(CC) vecsum4$(CEXT) $(LDLIBS) -o vecsum4$(OEXT)
	$(CC) vecsum8$(CEXT) $(LDLIBS) -o vecsum8$(OEXT)
	$(CC) vecsum4by4$(CEXT) $(LDLIBS) -o vecsum4by4$(OEXT)
	$(CC) vecsum4by2$(CEXT) $(LDLIBS) -o vecsum4by2$(OEXT)
	$(CC) vecsmooth$(CEXT) $(LDLIBS) -o vecsmooth$(OEXT)
	$(CC) vecsmooth_v4$(CEXT) $(LDLIBS) -o vecsmooth_v4$(OEXT)
	$(CC) vecsmooth_lmem$(CEXT) $(LDLIBS) -o vecsmooth_lmem$(OEXT)
	$(CC) matinit$(CEXT) $(LDLIBS) -o matinit$(OEXT)
	$(CC) transpose$(CEXT) $(LDLIBS) -o transpose$(OEXT)
	$(CC) transpose_lmem$(CEXT) $(LDLIBS) -o transpose_lmem$(OEXT)

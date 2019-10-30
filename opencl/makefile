CC=gcc
LDLIBS=-lOpenCL
CEXT=.c
OEXT=.out

TARGETS= vecsum$(CEXT) vecsum4$(CEXT) vecsum8$(CEXT) vecsum4by4$(CEXT) vecsum4by2$(CEXT) vecsmooth$(CEXT)
TARGETS+= vecsmooth_lmem$(CEXT) 


all: $(TARGETS)
	$(CC) vecsum$(CEXT) $(LDLIBS) -o vecsum$(OEXT)
	$(CC) vecsum4$(CEXT) $(LDLIBS) -o vecsum4$(OEXT)
	$(CC) vecsum8$(CEXT) $(LDLIBS) -o vecsum8$(OEXT)
	$(CC) vecsum4by4$(CEXT) $(LDLIBS) -o vecsum4by4$(OEXT)
	$(CC) vecsum4by2$(CEXT) $(LDLIBS) -o vecsum4by2$(OEXT)
	$(CC) vecsmooth$(CEXT) $(LDLIBS) -o vecsmooth$(OEXT)
	$(CC) vecsmooth_lmem$(CEXT) $(LDLIBS) -o vecsmooth_lmem$(OEXT)

CC=gcc
LDLIBS=-lOpenCL
CEXT=.c
OEXT=.out

all: vecsum$(CEXT) vecsum4$(CEXT) vecsum8$(CEXT)
	$(CC) vecsum$(CEXT) $(LDLIBS) -o vecsum$(OEXT)
	$(CC) vecsum4$(CEXT) $(LDLIBS) -o vecsum4$(OEXT)
	$(CC) vecsum8$(CEXT) $(LDLIBS) -o vecsum8$(OEXT)
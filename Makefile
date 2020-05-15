CXX = mpic++
CXXFLAGS = -std=c++11 -Wall -Wextra -Wshadow -Werror -O3 -DNDEBUG -ffast-math

INCLUDES =
LDFLAGS =
LIBS =

# likwid
#CXXFLAGS += -DUSE_LIKWID -pthread
#INCLUDES += -I/usr/local/likwid-3.1.2/include/
#LDFLAGS += -L/usr/local/likwid-3.1.2/lib/

TARGET = cg
OBJS = $(TARGET).o

all: $(TARGET)

$(TARGET): $(OBJS) Makefile
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS) $(LIBS)

$(TARGET).o: $(TARGET).cpp Timer.h Makefile
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $(TARGET).cpp

clean:
	@$(RM) -rf *.o $(TARGET)
